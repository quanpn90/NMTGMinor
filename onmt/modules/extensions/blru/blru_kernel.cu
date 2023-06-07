// Author: Christian Huber, KIT, 23.05.2023 - 26.05.2023

#include <torch/torch.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

//typedef float DTYPE_BASE; // compile with this for fp32
typedef c10::Half DTYPE_BASE; // compile with this for fp16

typedef c10::complex<DTYPE_BASE> DTYPE;

int64_t nextPowerOfTwo(int64_t n) {
    // Set all the bits after the most significant bit to 1
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++; // Increase the number by 1 to get the next power of two
    return n;
}

#define LOG_NUM_BANKS 4 // device with 2**LOG_NUM_BANKS filterbanks
#define CONFLICT_FREE_OFFSET(n) ((n)>>LOG_NUM_BANKS)

__device__ __forceinline__ void set(DTYPE* temp, int index, DTYPE value) {
    temp[index + CONFLICT_FREE_OFFSET(index)] = value;
}
__device__ __forceinline__ void add(DTYPE* temp, int index, DTYPE value) {
    temp[index + CONFLICT_FREE_OFFSET(index)] += value;
}
__device__ __forceinline__ DTYPE get(DTYPE* temp, int index) {
    return temp[index + CONFLICT_FREE_OFFSET(index)];
}

template <int n, bool reverse>
__global__ void lru_kernel(
    torch::PackedTensorAccessor32<DTYPE,2,torch::RestrictPtrTraits> Lambda_exp,
    torch::PackedTensorAccessor32<DTYPE,3,torch::RestrictPtrTraits> Bu,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> lengths,
    torch::PackedTensorAccessor32<DTYPE,3,torch::RestrictPtrTraits> out) {

    int length = lengths[blockIdx.x];
    extern __shared__ DTYPE temp[];

    int offset = 1;
    int i = 0;
    int thid = threadIdx.x;

    if(!reverse) {
        if(thid < length) {
            set(temp, thid, Bu[blockIdx.x][blockIdx.y][thid]);
        }
        if(blockDim.x+thid < length) {
            set(temp, blockDim.x+thid, Bu[blockIdx.x][blockIdx.y][blockDim.x+thid]);
        }
    }
    else {
        if(thid < length) {
            set(temp, length-1-thid, Bu[blockIdx.x][blockIdx.y][thid]);
        }
        if(blockDim.x+thid < length) {
            set(temp, length-1-blockDim.x-thid, Bu[blockIdx.x][blockIdx.y][blockDim.x+thid]);
        }
    }

    #pragma unroll
    for(int d = n>>1; d > 1; d >>= 1) {
        __syncthreads();
        if(thid+1 < d) {
            DTYPE lambda = Lambda_exp[blockIdx.y][i];
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            add(temp, bi, lambda * get(temp, ai));
        }
        offset <<= 1;
        i++;
    }

    if(thid==0) {
        set(temp, n-1, DTYPE(0,0));
    }

    #pragma unroll
    for(int d = 1; d < n; d <<= 1) {
        __syncthreads();
        if(thid < d) {
            DTYPE lambda = Lambda_exp[blockIdx.y][i];
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            DTYPE t = get(temp, ai);
            set(temp, ai, get(temp, bi));
            set(temp, bi, lambda * get(temp, bi) + t);
        }
        offset >>= 1;
        i--;
    }
    __syncthreads();

    if(!reverse) {
        if(thid < length) {
            out[blockIdx.x][blockIdx.y][thid] = get(temp, thid+1);
        }
        if(blockDim.x+thid < length) {
            out[blockIdx.x][blockIdx.y][blockDim.x+thid] = get(temp, blockDim.x+thid+1);
        }
    }
    else {
        if(thid < length) {
            out[blockIdx.x][blockIdx.y][thid] = get(temp, length-thid);

         }
        if(blockDim.x+thid < length) {
            out[blockIdx.x][blockIdx.y][blockDim.x+thid] = get(temp, length-blockDim.x-thid);
        }
    }
}

#define k(n) lru_kernel<n,reverse><<<blocks, n/2, sizeof(DTYPE) * (n+CONFLICT_FREE_OFFSET(n))>>>(\
        Lambda_exp.packed_accessor32<DTYPE,2,torch::RestrictPtrTraits>(),\
        Bu.packed_accessor32<DTYPE,3,torch::RestrictPtrTraits>(),\
        lengths.packed_accessor32<int,1,torch::RestrictPtrTraits>(),\
        out.packed_accessor32<DTYPE,3,torch::RestrictPtrTraits>());

template <bool reverse>
void generalized_cumsum(
    torch::Tensor Lambda_exp, // N x log(L)
    torch::Tensor Bu, // B x N x L
    torch::Tensor lengths, // B
    torch::Tensor out) { // B x N x L

    dim3 blocks(Bu.size(0),Bu.size(1));
    int64_t n = nextPowerOfTwo(Bu.size(2));

    switch(n) {
        case 2:
            k(2); break;
        case 4:
            k(4); break;
        case 8:
            k(8); break;
        case 16:
            k(16); break;
        case 32:
            k(32); break;
        case 64:
            k(64); break;
        case 128:
            k(128); break;
        case 256:
            k(256); break;
        case 512:
            k(512); break;
        case 1024:
            k(1024); break;
        case 2048:
            k(2048); break;
        default:
            assert(false);
    }
}

torch::Tensor blru_forward(
    torch::Tensor Lambda_exp, // N x log(L)
    torch::Tensor Bu, // B x N x L
    torch::Tensor lengths, // B
    int direction) { // 0: left-to-right, 1: right-to-left, 2: first half left-to-right, second half right-to-left

    CHECK_INPUT(Lambda_exp);
    CHECK_INPUT(Bu);
    CHECK_INPUT(lengths);

    auto out = torch::empty_like(Bu);

    if(direction==0) {
        generalized_cumsum<false>(Lambda_exp,Bu,lengths,out);
    } else if(direction==1) {
        generalized_cumsum<true>(Lambda_exp,Bu,lengths,out);
    } else if(direction==2) {
        generalized_cumsum<false>(Lambda_exp.slice(0,0,Bu.size(1)/2),
                                  Bu.slice(1,0,Bu.size(1)/2),
                                  lengths,
                                  out.slice(1,0,Bu.size(1)/2));
        generalized_cumsum<true>(Lambda_exp.slice(0,Bu.size(1)/2,Bu.size(1)),
                                 Bu.slice(1,Bu.size(1)/2,Bu.size(1)),
                                 lengths,
                                 out.slice(1,Bu.size(1)/2,Bu.size(1)));
    } else {
        assert(false);
    }

    return out;
}


std::vector<torch::Tensor> blru_backward(
    torch::Tensor grad_output, // B x N x L
    torch::Tensor Lambda_exp, // N x log(L)
    torch::Tensor output, // B x N x L
    torch::Tensor lengths, // B
    int direction) {

    CHECK_INPUT(Lambda_exp);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(lengths);
    CHECK_INPUT(output);

    //auto d_Bu = torch::empty_like(grad_output);
    auto d_Bu = torch::zeros_like(grad_output);
    torch::Tensor d_Lambda;

    if(direction==0) {
        generalized_cumsum<true>(Lambda_exp,grad_output,lengths,d_Bu);

        //auto tmp = torch::empty_like(output);
        auto tmp = torch::zeros_like(output);
        generalized_cumsum<false>(Lambda_exp,output,lengths,tmp);

        tmp = tmp.to(at::kComplexFloat);
        grad_output = grad_output.to(at::kComplexFloat);
        d_Lambda = torch::einsum("bnl,bnl->n", {grad_output.slice(2,1,grad_output.size(2)), tmp.slice(2,0,tmp.size(2)-1)});
    } else if(direction==1) {
        generalized_cumsum<false>(Lambda_exp,grad_output,lengths,d_Bu);

        //auto tmp = torch::empty_like(output);
        auto tmp = torch::zeros_like(output);
        generalized_cumsum<true>(Lambda_exp,output,lengths,tmp);
        tmp = tmp.to(at::kComplexFloat);
        grad_output = grad_output.to(at::kComplexFloat);
        d_Lambda = torch::einsum("bnl,bnl->n", {grad_output.slice(2,0,grad_output.size(2)-1), tmp.slice(2,1,tmp.size(2))});
    } else if(direction==2) {
        generalized_cumsum<true>(Lambda_exp.slice(0,0,d_Bu.size(1)/2),
                                 grad_output.slice(1,0,d_Bu.size(1)/2),
                                 lengths,
                                 d_Bu.slice(1,0,d_Bu.size(1)/2));
        generalized_cumsum<false>(Lambda_exp.slice(0,d_Bu.size(1)/2,d_Bu.size(1)),
                                  grad_output.slice(1,d_Bu.size(1)/2,d_Bu.size(1)),
                                  lengths,
                                  d_Bu.slice(1,d_Bu.size(1)/2,d_Bu.size(1)));

        grad_output = grad_output.to(at::kComplexFloat);

        //auto tmp = torch::empty_like(output.slice(1,0,d_Bu.size(1)/2));
        auto tmp = torch::zeros_like(output.slice(1,0,d_Bu.size(1)/2));
        generalized_cumsum<false>(Lambda_exp.slice(0,0,d_Bu.size(1)/2),
                                  output.slice(1,0,d_Bu.size(1)/2),
                                  lengths,
                                  tmp);
        auto tmp2 = tmp.to(at::kComplexFloat);
        auto d_Lambda_1 = torch::einsum("bnl,bnl->n", {grad_output.slice(1,0,d_Bu.size(1)/2).slice(2,1,grad_output.size(2)),
                                                       tmp2.slice(2,0,tmp2.size(2)-1)});

        tmp.zero_();
        generalized_cumsum<true>(Lambda_exp.slice(0,d_Bu.size(1)/2,d_Bu.size(1)),
                                 output.slice(1,d_Bu.size(1)/2,d_Bu.size(1)),
                                 lengths,tmp);

        tmp2 = tmp.to(at::kComplexFloat);
        auto d_Lambda_2 = torch::einsum("bnl,bnl->n", {grad_output.slice(1,d_Bu.size(1)/2,d_Bu.size(1)).slice(2,0,grad_output.size(2)-1),
                                                       tmp2.slice(2,1,tmp2.size(2))});
        d_Lambda = torch::cat({d_Lambda_1,d_Lambda_2});
    } else {
        assert(false);
    }

    return {d_Lambda, d_Bu};
}
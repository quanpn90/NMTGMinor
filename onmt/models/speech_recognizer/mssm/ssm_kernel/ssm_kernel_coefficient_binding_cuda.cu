#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <THC/THCAtomics.cuh>

#define CHECK_DEVICE(x) \
  TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                           \
  TORCH_CHECK(                                        \
      x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
      #x " must have shape (" #__VA_ARGS__ ")")
#define REDUCE_THREADS 128

#define REDUCE_THREADS_FWD 32


#define REDUCE_THREADS_A 512
#define REDUCE_THREADS_B 64
#define REDUCE_THREADS_C 1024




template <typename T, size_t N>
using CudaAcsr = at::GenericPackedTensorAccessor<T, N, at::RestrictPtrTraits, int32_t>;

template <int NUM_THREADS, typename scalar_t>
__global__ void kernel_coefficient_forward_kernel(
    CudaAcsr<scalar_t, 4> a,
    CudaAcsr<scalar_t, 3> b,
    CudaAcsr<scalar_t, 2> c,
    CudaAcsr<scalar_t, 4> out,
    int N,
    int H
) {

    __shared__ char shared_ch[NUM_THREADS * sizeof(scalar_t)];
    scalar_t* shared = (scalar_t*)&shared_ch;
    __shared__ char shared_b_elem_ch[sizeof(scalar_t)];
    scalar_t* shared_b_elem = (scalar_t*)&shared_b_elem_ch;

    int ic = blockIdx.x;
    int l = blockIdx.y;
    int qh = blockIdx.z;
    int q = qh / H;
    int h = qh % H;

    if (threadIdx.x == 0) {shared_b_elem[0] = b[q][l][h];}
    __syncthreads();

    scalar_t val = scalar_t(0.0);
    for (int i = threadIdx.x; i < N; i += NUM_THREADS) {
      val += a[ic][q][h][i] / (shared_b_elem[0] - c[q][i]);
    }
    shared[threadIdx.x] = val;
    __syncthreads();
    val = at::native::cuda_utils::BlockReduceSum<scalar_t>(val, shared);
    if (threadIdx.x == 0) {out[ic][q][l][h] = val;}
}

template <typename T>
torch::Tensor
kernel_coefficient_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor out) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  CHECK_DEVICE(c);
  const auto IC = a.size(0);
  const auto Q = a.size(1); // num heads
  const auto H = a.size(2); // input dim
  const auto N = a.size(3); // hidden dim
  const auto L = b.size(1); // seq length
  CHECK_SHAPE(b, Q, L, H);
  CHECK_SHAPE(c, Q, N);

  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = c10::complex<T>;
  const auto a_p = a.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>();
  const auto b_p = b.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
  const auto c_p = c.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  auto out_p = out.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>();
  dim3 grid(IC, L, Q * H);
  dim3 block(REDUCE_THREADS_FWD);
  kernel_coefficient_forward_kernel<REDUCE_THREADS_FWD, scalar_t>
      <<<grid, block, 0, stream>>>(a_p, b_p, c_p, out_p, N, H);

  return out;
}

template <int NUM_THREADS, typename scalar_t>
__global__ void
_abc_kernel(
    CudaAcsr<scalar_t, 4> a,
    CudaAcsr<scalar_t, 3> b,
    CudaAcsr<scalar_t, 2> c,
    CudaAcsr<scalar_t, 4> dout,
    CudaAcsr<scalar_t, 4> da,
    CudaAcsr<scalar_t, 3> db,
    CudaAcsr<scalar_t, 2> dc,
    int Q,
    int IC,
    int L,
    int H,
    int N,
    int G /* = IC * H * N + L * H + N */) {
  __shared__ char sh_array_ch[NUM_THREADS * sizeof(scalar_t)];
  scalar_t* sh_array = (scalar_t*)&sh_array_ch;
  __shared__ char sh_elem_ch[sizeof(scalar_t)];
  scalar_t* sh_elem = (scalar_t*)&sh_elem_ch;
  /* Here, G = IC * H * N + L * H + N
    In total, there are G * Q jobs
    (blockIdx.x, blockIdx.x) to prepare an idx in [0, G - 1]
    blockIdx.z = Q
  */
  int my_idx = blockIdx.x * L + blockIdx.y;
  /* [0, N - 1] for dc; let sa = N
    [sa, sa + IC * H * N - 1] for da; let sb = sa + IC * H * N
    [sb, sb + L * H - 1] for db
  */
  if (my_idx >= G)
    return;
  int sa = N;
  int sb = sa + IC * H * N;
  scalar_t val = scalar_t(0.0);
  int q = blockIdx.z;
  int tot;
  if (my_idx < sa) {
    // dc = sum_{ic, h, l} a/[(b-c)*]^2
    int n = my_idx % N;
    tot = IC * H * L;
    if (threadIdx.x == 0) {
      sh_elem[0] = c[q][n];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < tot; i += NUM_THREADS) {
      int ic = i % IC;
      int hl = i / IC;
      int h = hl % H;
      int l = hl / H;
      scalar_t diff_conj_inv =
          scalar_t(1.0) / std::conj(b[q][l][h] - sh_elem[0]);
      val += dout[ic][q][l][h] * a[ic][q][h][n] * diff_conj_inv * diff_conj_inv;
    }
    sh_array[threadIdx.x] = val;
    __syncthreads();
    val = at::native::cuda_utils::BlockReduceSum<scalar_t>(val, sh_array);
    if (threadIdx.x == 0) {
      dc[q][n] = val;
    }
  } else if (my_idx < sb) {
    // da = sum_{l} 1/[(b-c)*]
    tot = L;
    my_idx -= sa;
    int ic = my_idx % IC;
    int h_n = my_idx / IC;
    int h = h_n % H;
    int n = h_n / H;
    if (threadIdx.x == 0) {
      sh_elem[0] = c[q][n];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < tot; i += NUM_THREADS) {
      scalar_t diff_conj_inv =
          scalar_t(1.0) / std::conj(b[q][i][h] - sh_elem[0]);
      val += dout[ic][q][i][h] * diff_conj_inv;
    }
    sh_array[threadIdx.x] = val;
    __syncthreads();
    val = at::native::cuda_utils::BlockReduceSum<scalar_t>(val, sh_array);
    if (threadIdx.x == 0) {
      da[ic][q][h][n] = val;
    }
  } else {
    // db = sum_{ic, n} -a/[(b-c)*]^2
    tot = IC * N;
    my_idx -= sb;
    int l = my_idx % L;
    int h = my_idx / L;
    if (threadIdx.x == 0) {
      sh_elem[0] = b[q][l][h];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < tot; i += NUM_THREADS) {
      int ic = i % IC;
      int n = i / IC;
      scalar_t diff_conj_inv = scalar_t(1.0) / std::conj(sh_elem[0] - c[q][n]);
      val -= dout[ic][q][l][h] * a[ic][q][h][n] * diff_conj_inv * diff_conj_inv;
    }
    sh_array[threadIdx.x] = val;
    __syncthreads();
    val = at::native::cuda_utils::BlockReduceSum<scalar_t>(val, sh_array);
    if (threadIdx.x == 0) {
      db[q][l][h] = val;
    }
  }
}

template <int NUM_THREADS, typename scalar_t>
__global__ void kernel_coefficient_backward_a_kernel(
    CudaAcsr<scalar_t, 4> a,
    CudaAcsr<scalar_t, 3> b,
    CudaAcsr<scalar_t, 2> c,
    CudaAcsr<scalar_t, 4> dout,
    CudaAcsr<scalar_t, 4> da,
    int L,
    int H) {
  // da = sum_{l} 1/[(b-c)*]
  __shared__ char shared_ch[NUM_THREADS * sizeof(scalar_t)];
  scalar_t* shared = (scalar_t*)&shared_ch;
  __shared__ char shared_c_elem_ch[sizeof(scalar_t)];
  scalar_t* shared_c_elem = (scalar_t*)&shared_c_elem_ch;
  int ic = blockIdx.x;
  int q = blockIdx.y;
  int hn = blockIdx.z;
  int h = hn % H;
  int n = hn / H;
  if (threadIdx.x == 0) {
    shared_c_elem[0] = c[q][n];
  }
  __syncthreads();
  scalar_t val = scalar_t(0.0);
  for (int i = threadIdx.x; i < L; i += NUM_THREADS) {
    scalar_t diff_conj_inv =
        scalar_t(1.0) / std::conj(b[q][i][h] - shared_c_elem[0]);
    val += dout[ic][q][i][h] * diff_conj_inv;
  }
  shared[threadIdx.x] = val;
  __syncthreads();
  val = at::native::cuda_utils::BlockReduceSum<scalar_t>(val, shared);
  if (threadIdx.x == 0) {
    da[ic][q][h][n] = val;
  }
}

template <int NUM_THREADS, typename scalar_t>
__global__ void kernel_coefficient_backward_b_kernel(
    CudaAcsr<scalar_t, 4> a,
    CudaAcsr<scalar_t, 3> b,
    CudaAcsr<scalar_t, 2> c,
    CudaAcsr<scalar_t, 4> dout,
    CudaAcsr<scalar_t, 3> db,
    int N,
    int IC) {
  // db = sum_{ic, n} -a/[(b-c)*]^2
  __shared__ char shared_ch[NUM_THREADS * sizeof(scalar_t)];
  scalar_t* shared = (scalar_t*)&shared_ch;
  __shared__ char shared_b_elem_ch[sizeof(scalar_t)];
  scalar_t* shared_b_elem = (scalar_t*)&shared_b_elem_ch;
  int q = blockIdx.x;
  int l = blockIdx.y;
  int h = blockIdx.z;
  scalar_t val = scalar_t(0.0);
  if (threadIdx.x == 0) {
    shared_b_elem[0] = b[q][l][h];
  }
  __syncthreads();
  int tot = IC * N;
  for (int i = threadIdx.x; i < tot; i += NUM_THREADS) {
    int ic = i % IC;
    int n = i / IC;
    scalar_t diff_conj_inv =
        scalar_t(1.0) / std::conj(shared_b_elem[0] - c[q][n]);
    val -= dout[ic][q][l][h] * a[ic][q][h][n] * diff_conj_inv * diff_conj_inv;
  }
  shared[threadIdx.x] = val;
  __syncthreads();
  val = at::native::cuda_utils::BlockReduceSum<scalar_t>(val, shared);
  if (threadIdx.x == 0) {
    db[q][l][h] = val;
  }
}

template <int NUM_THREADS, typename scalar_t>
__global__ void kernel_coefficient_backward_c_kernel(
    CudaAcsr<scalar_t, 4> a,
    CudaAcsr<scalar_t, 3> b,
    CudaAcsr<scalar_t, 2> c,
    CudaAcsr<scalar_t, 4> dout,
    CudaAcsr<scalar_t, 2> dc,
    int IC,
    int H,
    int L) {
  // dc = sum_{ic, h, l} a/[(b-c)*]^2
  __shared__ char shared_ch[NUM_THREADS * sizeof(scalar_t)];
  scalar_t* shared = (scalar_t*)&shared_ch;
  __shared__ char shared_c_elem_ch[sizeof(scalar_t)];
  scalar_t* shared_c_elem = (scalar_t*)&shared_c_elem_ch;
  int q = blockIdx.x;
  int n = blockIdx.y;
  scalar_t val = scalar_t(0.0);
  int tot = IC * H * L;
  if (threadIdx.x == 0) {
    shared_c_elem[0] = c[q][n];
  }
  __syncthreads();
  // scalar_t c_kh = 0; // Kahan sum for precision
  for (int i = threadIdx.x; i < tot; i += NUM_THREADS) {
    int ic = i % IC;
    int hl = i / IC;
    int h = hl % H;
    int l = hl / H;
    scalar_t diff_conj_inv =
        scalar_t(1.0) / std::conj(b[q][l][h] - shared_c_elem[0]);
    /* scalar_t elem =
        dout[ic][q][l][h] * a[ic][q][h][n] * diff_conj_inv * diff_conj_inv -
        c_kh;
    scalar_t t_kh = val + elem;
    c_kh = (t_kh - val) - elem;
    val = t_kh; */
    val += dout[ic][q][l][h] * a[ic][q][h][n] * diff_conj_inv * diff_conj_inv;
  }
  shared[threadIdx.x] = val;
  __syncthreads();
  val = at::native::cuda_utils::BlockReduceSum<scalar_t>(val, shared);
  if (threadIdx.x == 0) {
    dc[q][n] = val;
  }
}

template <typename T>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
kernel_coefficient_backward(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor dout,
    torch::Tensor da,
    torch::Tensor db,
    torch::Tensor dc) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  CHECK_DEVICE(c);
  CHECK_DEVICE(dout);
  const auto IC = a.size(0);
  const auto Q = a.size(1); // num heads
  const auto H = a.size(2); // input dim
  const auto N = a.size(3); // hidden dim
  const auto L = b.size(1); // seq length
  CHECK_SHAPE(b, Q, L, H);
  CHECK_SHAPE(c, Q, N);
  CHECK_SHAPE(dout, IC, Q, L, H);

  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = c10::complex<T>;
  const auto a_p = a.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>();
  const auto b_p = b.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
  const auto c_p = c.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  const auto dout_p =
      dout.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>();
  auto da_p = da.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>();
  auto db_p = db.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
  auto dc_p = dc.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  dim3 grid_da(IC, Q, H * N);
  dim3 blocka(REDUCE_THREADS_A);
  dim3 grid_db(Q, L, H);
  dim3 blockb(REDUCE_THREADS_B);
  dim3 grid_dc(Q, N);
  dim3 blockc(REDUCE_THREADS_C);
  kernel_coefficient_backward_a_kernel<REDUCE_THREADS_A, scalar_t>
      <<<grid_da, blocka, 0, stream>>>(a_p, b_p, c_p, dout_p, da_p, L, H);
  kernel_coefficient_backward_b_kernel<REDUCE_THREADS_B, scalar_t>
      <<<grid_db, blockb, 0, stream>>>(a_p, b_p, c_p, dout_p, db_p, N, IC);
  kernel_coefficient_backward_c_kernel<REDUCE_THREADS_C, scalar_t>
      <<<grid_dc, blockc, 0, stream>>>(a_p, b_p, c_p, dout_p, dc_p, IC, H, L);
  /*int G = IC * H * N + L * H + N;
  dim3 grid((G + L - 1) / L, L, Q);
  dim3 block(REDUCE_THREADS);
  kernel_coefficient_backward_abc_kernel<REDUCE_THREADS, scalar_t>
      <<<grid, block, 0, stream>>>(
          a_p, b_p, c_p, dout_p, da_p, db_p, dc_p, Q, IC, L, H, N, G);*/
  return std::make_tuple(da, db, dc);
}


// Instantiate for floating point types
template torch::Tensor
kernel_coefficient_forward<float>(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor out);

template torch::Tensor
kernel_coefficient_forward<double>(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor out);



template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
kernel_coefficient_backward<float>(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor dout,
    torch::Tensor da,
    torch::Tensor db,
    torch::Tensor dc);


template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
kernel_coefficient_backward<double>(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor dout,
    torch::Tensor da,
    torch::Tensor db,
    torch::Tensor dc);


// int mlp_fp<float>(
//     float* X,
//     int input_features,
//     int batch_size,
//     float** WPtr,
//     float** BPtr,
//     int num_layers,
//     int* output_features,
//     float* Y,
//     float* reserved_space,
//     float* reserved_activations,
//     uint8_t* reserved_mask,
//     void* lt_workspace,
//     float p);

// PYBIND11_MODULE(ssm_kernel_coefficient_binding, m) {
//   m.def("kernel_coefficient_forward_float", &kernel_coefficient_forward<float>);
//   m.def(
//       "kernel_coefficient_forward_double", &kernel_coefficient_forward<double>);
//   m.def(
//       "kernel_coefficient_backward_float", &kernel_coefficient_backward<float>);
//   m.def(
//       "kernel_coefficient_backward_double",
//       &kernel_coefficient_backward<double>);
// }


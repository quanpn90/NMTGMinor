import torch
from torch.utils import cpp_extension
from setuptools import setup, find_packages
import subprocess

import sys
import warnings
import os

from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


cmdclass = {}
ext_modules = []

cmdclass['build_ext'] = BuildExtension.with_options(use_ninja=False)

# Check, if CUDA11 is installed for compute capability 8.0
cc_flag = []
_, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
# if int(bare_metal_major) >= 11:
#     cc_flag.append('-gencode')
#     cc_flag.append('arch=compute_80,code=sm_80')

cc_flag.append('-gencode')
cc_flag.append('arch=compute_75,code=sm_75')

# subprocess.run(["git", "submodule", "update", "--init", "cutlass"])
subprocess.run(["git", "clone", "https://github.com/NVIDIA/cutlass.git", "cutlass"])
# subprocess.run(["git", "-C", "cutlass", "checkout", "ed2ed4d667ce95e1371bd62db32b6a114e774336"])
subprocess.run(["git", "-C", "cutlass", "checkout", "fe3438a3c1ccbdd03dc1aca3bb68099a9e2a58bd"])


# ext_modules.append(
#     CUDAExtension(name='encdec_multihead_attn_cuda',
#                   sources=['encdec_multihead_attn.cpp',
#                            'encdec_multihead_attn_cuda.cu'],
#                   extra_compile_args={'cxx': ['-O3', ],
#                                       'nvcc': ['-O3',
#                                                '-I./cutlass/',
#                                                '-U__CUDA_NO_HALF_OPERATORS__',
#                                                '-U__CUDA_NO_HALF_CONVERSIONS__',
#                                                '--expt-relaxed-constexpr',
#                                                '--expt-extended-lambda',
#                                                '--use_fast_math'] + cc_flag}))
#
# ext_modules.append(
#     CUDAExtension(name='fused_dropout_add_cuda',
#                   sources=['dropout_add.cpp',
#                            'dropout_add_cuda.cu'],
#                   extra_compile_args={'cxx': ['-O3', ],
#                                       'nvcc': ['-O3',
#                                                '-U__CUDA_NO_HALF_OPERATORS__',
#                                                '-U__CUDA_NO_HALF_CONVERSIONS__',
#                                                '--expt-relaxed-constexpr',
#                                                '--expt-extended-lambda',
#                                                '--use_fast_math'] + cc_flag}))
#
#
# ext_modules.append(
#     CUDAExtension(name='mask_softmax_dropout_cuda',
#                   sources=['masked_softmax_dropout.cpp',
#                            'masked_softmax_dropout_cuda.cu'],
#                   extra_compile_args={'cxx': ['-O3', ],
#                                       'nvcc': ['-O3',
#                                                '-I./cutlass/include',
#                                                '-U__CUDA_NO_HALF_OPERATORS__',
#                                                '-U__CUDA_NO_HALF_CONVERSIONS__',
#                                                '--expt-relaxed-constexpr',
#                                                '--expt-extended-lambda',
#                                                '--use_fast_math'] + cc_flag}))

ext_modules.append(
    CUDAExtension(name='rel_self_attn_cuda',
                  sources=['relative_self_attn.cpp',
                           'relative_self_attn_cuda.cu'],
                  extra_compile_args={'cxx': ['-O3',],
                                      'nvcc':['-O3',
                                              '-I./cutlass/',
                                              '-U__CUDA_NO_HALF_OPERATORS__',
                                              '-U__CUDA_NO_HALF_CONVERSIONS__',
                                              '--expt-relaxed-constexpr',
                                              '--expt-extended-lambda',
                                              '--use_fast_math'] + cc_flag}))
# ext_modules.append(
#     CUDAExtension(name='fast_self_multihead_attn_bias',
#                   sources=['self_multihead_attn_bias.cpp',
#                            'self_multihead_attn_bias_cuda.cu'],
#                   extra_compile_args={'cxx': ['-O3',],
#                                       'nvcc':['-O3',
#                                               '-gencode', 'arch=compute_70,code=sm_70',
#                                               '-I./cutlass/',
#                                               '-U__CUDA_NO_HALF_OPERATORS__',
#                                               '-U__CUDA_NO_HALF_CONVERSIONS__',
#                                               '--expt-relaxed-constexpr',
#                                               '--expt-extended-lambda',
#                                               '--use_fast_math'] + cc_flag}))
# ext_modules.append(
#     CUDAExtension(name='fast_self_multihead_attn',
#                   sources=['self_multihead_attn.cpp',
#                            'self_multihead_attn_cuda.cu'],
#                   extra_compile_args={'cxx': ['-O3',],
#                                       'nvcc':['-O3',
#                                               '-gencode', 'arch=compute_70,code=sm_70',
#                                               '-I./cutlass/',
#                                               '-U__CUDA_NO_HALF_OPERATORS__',
#                                               '-U__CUDA_NO_HALF_CONVERSIONS__',
#                                               '--expt-relaxed-constexpr',
#                                               '--expt-extended-lambda',
#                                               '--use_fast_math'] + cc_flag}))


setup(
    name='optimized_multihead_attention',
    version='0.1',
    packages=find_packages(exclude=('build',
                                    'csrc',
                                    'include',
                                    'tests',
                                    'dist',
                                    'docs',
                                    'tests',
                                    'examples',
                                    'apex.egg-info',)),
    description='CUDA/C++ Pytorch extension for multi-head attention ported from NVIDIA apex',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

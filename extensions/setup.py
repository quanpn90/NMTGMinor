import torch
from torch.utils import cpp_extension
from setuptools import setup, find_packages
import subprocess

import sys
import warnings
import os
from datetime import datetime

from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")
    # print(bare_metal_minor, bare_metal_major)

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        print("Cuda extensions are being compiled with a version of Cuda that does " +
              "not match the version used to compile Pytorch binaries.  " +
              "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda) +
              "In some cases, a minor-version mismatch will not cause later errors:  " +
              "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
              "You can try commenting out this check (at your own risk).")

    return int(bare_metal_minor), int(bare_metal_major)


# Check, if ATen/CUDAGenerator.h is found, otherwise use the new
# ATen/CUDAGeneratorImpl.h, due to breaking change in https://github.com/pytorch/pytorch/pull/36026
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]


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

cc_flag = []
print(cpp_extension.CUDA_HOME)
_, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)

cc_flag.append('-gencode')
cc_flag.append('arch=compute_70,code=sm_70')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_75,code=sm_75')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_80,code=sm_80')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_86,code=sm_86')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_90,code=sm_90')

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ['-DVERSION_GE_1_1']
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ['-DVERSION_GE_1_3']
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ['-DVERSION_GE_1_5']
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

# subprocess.run(["git", "submodule", "update", "--init", "cutlass"])
# subprocess.run(["git", "clone", "https://github.com/NVIDIA/cutlass.git", "multihead_attn/cutlass"])
# subprocess.run(["git", "-C", "cutlass", "checkout", "ed2ed4d667ce95e1371bd62db32b6a114e774336"])
# subprocess.run(["git", "-C", "cutlass", "checkout", "fe3438a3c1ccbdd03dc1aca3bb68099a9e2a58bd"])


bare_metal_minor, bare_metal_major = check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)
print("GENERATOR FLAG:", generator_flag)

ext_modules.append(
    CUDAExtension(name='fast_layer_norm_cuda',
                  sources=['layer_norm/ln_api.cpp',
                           'layer_norm/ln_fwd_cuda_kernel.cu',
                           'layer_norm/ln_bwd_semi_cuda_kernel.cu'],
                  include_dirs=[os.path.join(this_dir, 'include')],
                  extra_compile_args={'cxx': ['-O3'] + ['-march=core-avx2'] + version_dependent_macros + generator_flag,
                                      'nvcc': ['-O3',
                                               '-U__CUDA_NO_HALF_OPERATORS__',
                                               '-U__CUDA_NO_HALF_CONVERSIONS__',
                                               '-U__CUDA_NO_BFLOAT16_OPERATORS__',
                                               '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                                               '-U__CUDA_NO_BFLOAT162_OPERATORS__',
                                               '-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
                                               '--expt-relaxed-constexpr',
                                               '--expt-extended-lambda',
                                               '--use_fast_math'] + cc_flag
                                               + version_dependent_macros + generator_flag}))


xentropy_ver = datetime.today().strftime("%y.%m.%d")
print(f"`--xentropy` setting version of {xentropy_ver}")
ext_modules.append(
    CUDAExtension(name='xentropy_cuda',
                  sources=['xentropy/interface.cpp',
                           'xentropy/xentropy_kernel.cu'],
                  include_dirs=[os.path.join(this_dir, 'include')],
                  extra_compile_args={"cxx": ["-O3"] + version_dependent_macros + [f'-DXENTROPY_VER="{xentropy_ver}"'],
                                      'nvcc': ['-O3'] + version_dependent_macros}))




setup(
    name='extra_cuda_modules',
    version='0.1',
    description='CUDA/C++ Pytorch extension for layer norm and xentropy ported from NVIDIA apex',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
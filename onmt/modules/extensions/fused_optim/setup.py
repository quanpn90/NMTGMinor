import os
import torch
from torch.utils import cpp_extension
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

cc_flag = []
ext_modules = []
cc_flag.append('-gencode')
cc_flag.append('arch=compute_75,code=sm_75')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_80,code=sm_80')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_86,code=sm_86')

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


ext_modules.append(
            CUDAExtension(name='fused_optim',
                          sources=['frontend.cpp',
                                   'multi_tensor_adam.cu'],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc':['-lineinfo',
                                                      '-O3',
                                                      # '--resource-usage',
                                                      '--use_fast_math'] + version_dependent_macros}))

setup(
    name="fused_optim",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
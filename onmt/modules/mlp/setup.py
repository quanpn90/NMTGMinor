from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="cuda_fused_mlp",
    ext_modules=[
        CUDAExtension(name='fused_mlp',
                      sources=['mlp.cpp',
                               'mlp_cuda.cu'],
                      extra_compile_args={'cxx': ['-O3'],
                                          'nvcc':['-O3']})
    ],
    cmdclass={"build_ext": BuildExtension},
)

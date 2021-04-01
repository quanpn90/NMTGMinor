from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="mlp_onmt",
    ext_modules=[
        CUDAExtension(name='mlp_onmt',
                      sources=['mlp.cpp',
                               'mlp_cuda.cu'],
                      extra_compile_args={'cxx': ['-O3'],
                                          'nvcc':['-O3']})
    ],
    cmdclass={"build_ext": BuildExtension},
)

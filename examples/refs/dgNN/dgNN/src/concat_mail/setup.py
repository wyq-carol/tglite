from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='concat_mail',
    ext_modules=[
        CUDAExtension(
            name='concat_mail',
            sources=[
                'concat_mail.cpp',
                'concat_mail_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-fopenmp', '-std=c++17'],
                'nvcc': [
                    '-g',
                    '-lineinfo',
                    '-O2',
                    '-arch=sm_80',
                    '-std=c++17',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    #'-maxrregcount=32'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    #install_requires=['torch']
)
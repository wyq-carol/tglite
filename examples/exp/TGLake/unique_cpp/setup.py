from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='custom_unique',
    ext_modules=[CppExtension('custom_unique', ['custom_unique.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)
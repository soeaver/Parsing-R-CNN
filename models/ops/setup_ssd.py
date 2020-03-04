from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension
from setuptools import setup

import numpy as np


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [
    Extension(
        name='cython_bbox',
        sources=['cython_bbox.pyx'],
        extra_compile_args=['-Wno-cpp'],
        include_dirs=[numpy_include]
    ),
    Extension(
        name='cython_nms',
        sources=['cython_nms.pyx'],
        extra_compile_args=['-Wno-cpp'],
        include_dirs=[numpy_include]
    )
]

setup(
    name='pet',
    ext_modules=cythonize(ext_modules)
)

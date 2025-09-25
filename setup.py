# python setup.py build_ext --inplace

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="ray_functions",
        sources=["ray_functions.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    ext_modules=cythonize(extensions),
)

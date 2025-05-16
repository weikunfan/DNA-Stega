from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "stega_cy",
        ["stega_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        language="c++"
    )
]

setup(
    ext_modules=cythonize(extensions)
) 
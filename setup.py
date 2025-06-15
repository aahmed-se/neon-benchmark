
from setuptools import setup, Extension
import pybind11
import sys
import platform

# Detect if we're on macOS with Apple Silicon
is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

# Base compiler arguments
extra_compile_args = ['-std=c++23', '-O3']
extra_link_args = ['-std=c++23']

# Add specific flags for Apple Silicon
if is_apple_silicon:
    extra_compile_args.extend(['-arch', 'arm64'])
    extra_link_args.extend(['-arch', 'arm64'])

ext_modules = [
    Extension(
        "neon_ops",
        ["neon_ops.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="neon_ops",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.10.0'],
    python_requires='>=3.8',
    setup_requires=['pybind11>=2.10.0'],
)
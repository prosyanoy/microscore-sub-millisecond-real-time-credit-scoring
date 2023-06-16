import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

flags = ["-O3", "-fno-math-errno"]
if os.environ.get("MICROSCORE_NATIVE") == "1":  # opt-in: binary is then not portable
    flags.append("-march=native")

ext = Pybind11Extension(
    "microscore._treeeval",
    ["cpp/treeeval.cpp"],
    cxx_std=17,
    extra_compile_args=flags,
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})

# -*- coding: utf-8 -*-

from setuptools import setup, Extension
import numpy as np
import os, sys


# Check to see if there's a prefered compiler on this machine
if "CC" in os.environ:
  print("*"*110)
  print(f"Detected env variable: CC={os.environ['CC']}")
  print(f"We'll attempt to compile this library with {os.environ['CC']}.")
  print("If compilation fails, remove this variable from your environment by typing 'unset CC', and try running setup again.")
  print("*"*110)


# We have to extend BuildExt to access compiler flags, etc
# See distutils/ccompiler.py (Yes, the new setuptools still depends on the old distutils.)
from setuptools.command.build_ext import build_ext
class BuildExt(build_ext):
  def build_extensions(self):
    cc = os.environ["CC"] if "CC" in os.environ else None
    if sys.platform == 'darwin':
      if os.system("which clang")==0:
        cc = "clang"
        self.compiler.compiler_so.append('-stdlib=libc++')
        self.compiler.compiler_so.remove('-Wstrict-prototypes') # gets rid of a useless warning
    elif sys.platform == 'linux':
      cc = "g++"
    if cc is not None:
      self.compiler.compiler_so[0] = cc
      self.compiler.compiler[0] = cc
      self.compiler.linker_so[0] = cc
    super(BuildExt, self).build_extensions()


cpp_module = Extension(
  'gdtw/gdtwcpp', 
  sources=['gdtw/gdtw_solver.cpp','gdtw/utils.cpp','gdtw/numpyobject.cpp'],
  include_dirs=[np.get_include()],
  extra_compile_args=["-Ofast", "-Wall", "-std=c++11"],
  language="c++11"
)

setup_params = setup(
  cmdclass={'build_ext': BuildExt},
  setup_requires=["numpy"],
  install_requires=["numpy"],
  ext_modules=[cpp_module],
  packages=['gdtw'],
  name='gdtw',
  version='1.0',
  description='General Dynamic Time Warping',
  author='Dave Deriso',
  author_email='dderiso@alumni.stanford.edu',
  url='https://dderiso.github.io/gdtw_staging',
  long_description='GDTW is a Python/C++ library that performs dynamic time warping. It is based on the paper by Dave Deriso and Stephen Boyd.'
)


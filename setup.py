# SPDX-License-Identifier: Apache-2.0
# 
# Copyright (C) 2019-2024 Dave Deriso <dderiso@alumni.stanford.edu>, Twitter: @davederiso
# Copyright (C) 2019-2024 Stephen Boyd
# 
# GDTW is a Python/C++ library that performs dynamic time warping.
# GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization, 
# which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters. 
# 
# Paper: https://rdcu.be/cT5dD
# Source: https://github.com/dderiso/gdtw
# Docs: https://dderiso.github.io/gdtw

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

from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
  def build_extensions(self):
    cc = None
    if "CC" in os.environ:
      cc = os.environ["CC"]
    elif sys.platform == 'darwin':
      if os.system("which g++")==0:
        cc = "g++"
    elif sys.platform == "linux":
      cpp_compiler = "g++"
    if cc is not None:
      self.compiler.compiler_so[0] = cc
      self.compiler.compiler[0]    = cc
      self.compiler.linker_so[0]   = cc

      self.compiler.compiler_so.append('-stdlib=libc++')
      self.compiler.compiler.append('-stdlib=libc++')
    super(BuildExt, self).build_extensions()

cpp_module = Extension(
  'gdtw/gdtwcpp', 
  sources=['gdtw/gdtw_solver.cpp'],
  include_dirs=[
    np.get_include()
  ],
  extra_compile_args=["-Ofast", "-Wall", "-std=c++11", "-flto", "-march=native"],
  language="c++11"
)

with open("readme.md","r") as f:
  long_description = f.read();

setup_params = setup(
  name='gdtw',
  version='1.0.7',
  author='Dave Deriso',
  author_email='dderiso@alumni.stanford.edu',
  description='General Dynamic Time Warping',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://dderiso.github.io/gdtw',
  project_urls={
    "Bug Tracker": "https://github.com/dderiso/gdtw/issues",
  },
  classifiers=[
    "Programming Language :: Python :: 3",
    "Programming Language :: C++",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent"
  ],
  cmdclass={'build_ext': BuildExt},
  setup_requires=["numpy"],
  install_requires=["numpy"],
  ext_modules=[cpp_module],
  packages=['gdtw']
)


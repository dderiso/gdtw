# SPDX-License-Identifier: Apache-2.0
# 
# Copyright (C) 2019-2023 Dave Deriso <dderiso@alumni.stanford.edu>, Twitter: @davederiso
# Copyright (C) 2019-2023 Stephen Boyd
# 
# GDTW is a Python/C++ library that performs dynamic time warping.
# GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization, 
# which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters. 
# 
# Paper: https://rdcu.be/cT5dD
# Source: https://github.com/dderiso/gdtw
# Docs: https://dderiso.github.io/gdtw

from .gdtw import GDTW
from .warp import warp
from .gdtwcpp import solve
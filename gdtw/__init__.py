# SPDX-License-Identifier: Apache-2.0
# 
# Copyright (C) 2019-2023 Dave Deriso <dderiso@alumni.stanford.edu>
# Copyright (C) 2019-2023 Stephen Boyd
# 
# GDTW is a Python/C++ library that performs dynamic time warping.
# It is based on a paper by Dave Deriso and Stephen Boyd.
# GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization, 
# which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters. 
# 
# Visit: https://github.com/dderiso/gdtw (source)
# Visit: https://dderiso.github.io/gdtw  (docs)

from .gdtw import GDTW
from .warp import warp
from .gdtwcpp import solve
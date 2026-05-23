# SPDX-License-Identifier: Apache-2.0
# 
# Copyright (C) 2017-2026 
# Dave Deriso <dderiso@alumni.stanford.edu>, Twitter: @davederiso
# Stephen Boyd
# 
# GDTW is a Python/C++ library that performs dynamic time warping.
# GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization, 
# which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters. 
# 
# Paper: https://rdcu.be/cT5dD
# Source: https://github.com/dderiso/gdtw
# Docs: https://dderiso.github.io/gdtw

import numpy as np
from .utils import scale

'''
Signals can be specified in 3 ways:
1. A function
2. An array of samples: scalar shape (T,) or vector-valued shape (T, d)
3. A tuple: (array of samples, 1-D array of time points [t_1, ..., t_T])

In all cases, we'll return a sample array and a callable interpolator.
Multi-channel signals (d > 1) are interpolated per channel and stacked on a trailing axis.
'''

# Piecewise-linear interpolation, constant beyond boundaries (see "Signal" section in paper).
# Accepts 1-D z of shape (T,) or 2-D z of shape (T, d). Output gains a trailing channel axis
# only when z is 2-D, so 1-D callers see identical behavior.
def piecewise_linear_interpolate(t_z, z):
    z = np.asarray(z, dtype=np.double)
    if z.ndim == 1:
        return lambda t_: np.interp(t_, t_z, z)
    if z.ndim != 2:
        raise ValueError(f"Signal must be 1-D (T,) or 2-D (T, d); got shape {z.shape}.")
    def f(t_):
        return np.stack([np.interp(t_, t_z, z[:, k]) for k in range(z.shape[1])], axis=-1)
    return f

def memoize(f):
    cache = {}
    def memoized_f(arg):
        arg_hash = hash(arg.tobytes())
        if arg_hash not in cache:
            cache[arg_hash] = f(arg)
        return cache[arg_hash]
    return memoized_f

def array_to_memoized_f(t_x, x):
    x = np.asarray(x, dtype=np.double)
    if x.ndim == 1:
        def f(t_):
            return np.interp(t_, t_x, x)
    else:
        def f(t_):
            return np.stack([np.interp(t_, t_x, x[:, k]) for k in range(x.shape[1])], axis=-1)
    return memoize(f)

class Signal:
    def __init__(self, z, name, N, scale_signals=True, scale_range=[-1,1], verbose=0):
        self.scale_signals  = scale_signals
        self.scale_range    = scale_range
        self.N              = N
        self.z              = z
        self.name           = name
        self.verbose        = verbose
        self.check_signal()

    def check_signal(self):
        
        # Case 1: If our signal is a function,
        if callable(self.z):
            
            # we'll keep it that way, though we may scale it.
            self.z_f = self.z
            # Since we're not given an explicit time sequence, we'll construct one assuming evenly spaced samples,
            t_z = np.linspace(0,1,num=self.N).astype(np.double)
            # and generate an array.
            self.z_a = self.z_f(t_z)

            # If the user wants us to scale the signals, we'll have to test the range of the function.
            if self.scale_signals:
                # For 1-D signals: scalar min/max. For multi-D (T, d): per-channel min/max so a
                # quiet channel doesn't get rescaled by a louder one.
                z_sample = np.asarray(self.z_a, dtype=np.double)
                if z_sample.ndim == 1:
                    needs_scale = (z_sample.min() != self.scale_range[0]
                                   or z_sample.max() != self.scale_range[1])
                else:
                    mins = np.nanmin(z_sample, axis=0)
                    maxs = np.nanmax(z_sample, axis=0)
                    needs_scale = (not np.all(mins == self.scale_range[0])
                                   or not np.all(maxs == self.scale_range[1]))
                if needs_scale:
                    # we'll construct a new scaled signal (which may be inaccurate)
                    self.z_a = scale(z_sample, self.scale_range)
                    self.z_f = piecewise_linear_interpolate(t_z, self.z_a)
                    # and alert the user of this choice.
                    if self.verbose > 0:
                        print(f"Signal {self.name} is given as a function that not scaled to the desired range. We're scaling it for you (which may be inaccurate), or else you'll have to modify the function yourself so that it meets the desired range.")

        # Case 2: If z is a sequence,
        elif isinstance(self.z, np.ndarray) or isinstance(self.z, list):
            # then ensure it's a numpy array.
            self.z_a = np.array(self.z, dtype=np.double)
            # Reject higher-rank inputs up-front so downstream scale/interp don't produce NaNs.
            if self.z_a.ndim > 2:
                raise ValueError(
                    f"Signal {self.name} has shape {self.z_a.shape}; only 1-D (T,) or 2-D (T, d) are supported."
                )

            # If the user requests us to scale this signal (recommended to prevent numerical underflow),
            if self.scale_signals:
                # we'll scale the sequence to the desired range, given by the parameter self.scale_range.
                self.z_a = scale(self.z_a,self.scale_range)

            # Since we're not given an explicit time sequence, we'll construct one assuming evenly spaced samples,
            t_z = np.linspace(0,1,num=self.z_a.shape[0]).astype(np.double)
            # and alert the user of this asssumption.
            if self.verbose > 0:
                print(f"Assuming {self.name} is sampled at even intervals. If this is incorrect, please set t_{self.name} explicitly by passing a tuple containing (array of signal samples, array of time steps).")

            # Finally, we'll construct a function from our signal samples using piecewise linear interpolation.
            self.z_f = piecewise_linear_interpolate(t_z, self.z_a)

        # Case 3: If z is a tuple,
        elif isinstance(self.z, tuple):
            # then we would expect there to be only two elements: samples and time steps.
            if len(self.z) != 2:
                raise ValueError(f"Signal {self.name} is given as a tuple, but does not have 2 entries. A signal tuple should have the following form: (N-dimensional array of samples [x_1, x_2, ..., x_T], 1-dimensional array of time points [t_1, ..., t_T]).")
            # If that's true,
            else:
                # we'll unpack it
                self.z_a,t_z = self.z
                # and inspect the samples 
                if isinstance(self.z_a, np.ndarray) or isinstance(self.z_a, list):
                    # to ensure it's a numpy array.
                    self.z_a = np.array(self.z_a, dtype=np.double)
                # If the samples aren't given as a sequence, then we'll have to stop here.
                else:
                    raise ValueError(f"Signal {self.name} is given as a tuple, but the first entry is not an array.")

                # If the user requests us to scale this signal (recommended to prevent numerical underflow),
                if self.scale_signals:
                    # we'll scale the sequence to the desired range, given by the parameter self.scale_range.
                    self.z_a = scale(self.z_a,self.scale_range)

                # We'll also inspect the time steps 
                if isinstance(t_z, np.ndarray) or isinstance(t_z, list):
                    # to ensure it's a numpy array
                    t_z = np.array(t_z, dtype=np.double)
                    # and see if it's multidimensional. 
                    if t_z.ndim > 1:
                        # If so, we'll complain.
                        raise ValueError(f"Signal {self.name} is given as a tuple, but the second entry (time) is given as a multidimensional array. Time can only be 1-D.")

                # Finally, we'll construct a function from our signal samples using piecewise linear interpolation.
                self.z_f = piecewise_linear_interpolate(t_z, self.z_a)

        # If we're not given a signal, then we'll obviously throw an error.
        elif self.z is None:
            raise ValueError(f"Signal {self.name} is missing. Both signals (x and y) are required.")

        # Reject rank > 2 inputs early with a clear message; samples must be (T,) or (T, d).
        if self.z_a.ndim > 2:
            raise ValueError(
                f"Signal {self.name} has shape {self.z_a.shape}; only 1-D (T,) or 2-D (T, d) are supported."
            )
        # Record the channel count so downstream code can validate that x.d == y.d.
        self.d = 1 if self.z_a.ndim == 1 else self.z_a.shape[1]

        return self

    def get(self):
        return self.z_a, self.z_f, self.d

def signal(*args,**kwargs):
    return Signal(*args,**kwargs).get()

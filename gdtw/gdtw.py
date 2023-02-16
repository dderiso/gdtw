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



import numpy as np
import time
from .gdtwcpp import solve
from .signal import signal
from .utils import process_function

class GDTW:
    def __init__(self):
        # generic input vars
        self.x              = None
        self.x_a            = None
        self.x_f            = None
        self.y              = None
        self.y_a            = None
        self.y_f            = None

        # params and loss, regularizer functionals
        self.t              = None # t we integrate over
        self.lambda_cum     = 1
        self.lambda_inst    = .1
        self.Loss           = "L2"
        self.R_cum          = "L2"
        self.R_inst         = "L2"
        self.loss_f         = None

        # search space size
        self.N              = None
        self.N_default      = 300
        self.M              = None
        self.M_max          = 300
        self.eta            = .15
        
        # slope constraints and boundary conditions
        self.s_min          = 1e-8
        self.s_max          = 1e8
        self.s_beta         = 0
        self.BC_start_stop  = True

        # termination conditions
        self.max_iters      = 10
        self.epsilon_abs    = 1e-1
        self.epsilon_rel    = 1e-2  

        # for inspecting each iteration
        self.callback = False

        # misc.
        self.verbose        = 0
        self.uid            = None

        # private
        self.vectorized_Loss= False
        self.D              = None
        self.iteration      = 0 
        self.time_solve     = None
        self.f_tau_         = None

    def allocate(self):
        # graph
        self.Tau          = np.zeros((self.N,self.M),dtype=np.double)
        self.u            = None
        self.l            = None
        self.u_orig       = None
        self.l_orig       = None

        # solution
        self.tau          = np.zeros( self.N,        dtype=np.double) # phi is continuous
        self.path         = np.zeros( self.N,        dtype=np.int)    # path is discrete
        self.f_tau        = np.double(0.0)
        self.phi          = lambda t_: np.interp(t_, self.t, self.tau)

        return self

    def compute_taus(self):
        # initial u and ls
        if self.iteration == 0:
            self.u           = np.min([self.s_beta  + self.s_max*self.t,  self.s_beta + 1-self.s_min*(1-self.t) ],axis=0).astype(np.double)
            self.l           = np.max([               self.s_min*self.t, -self.s_beta + 1-self.s_max*(1-self.t) ],axis=0).astype(np.double)
            
            # restrict domain of phi to domain of t, since x and y may not be defined outside [t_min, t_max]
            self.u           = np.min([self.u,np.repeat(1,self.N)],axis=0)
            self.l           = np.max([self.l,np.repeat(0,self.N)],axis=0)
            self.u_orig      = self.u.copy()
            self.l_orig      = self.l.copy()

        # update u and l by factor eta, keep within bounds of original l and u
        else:
            tau_range = self.eta * (self.u-self.l)/np.double(2.)
            self.u = np.min([self.tau+tau_range,self.u_orig],axis=0)
            self.l = np.max([self.tau-tau_range,self.l_orig],axis=0)

        # compute taus for given an u and l
        a = np.stack((self.l,self.u-self.l),axis=1)
        b = np.vstack((np.ones(self.M),np.arange(self.M).astype(np.double)/np.double(self.M-1)))
        self.Tau = np.dot(a,b)
        # sanity check: this should decrease as u[i] shrinks towards phi[i] 
        # print( self.u[int(self.N/2)] - self.l[int(self.N/2)] )
        return self

    def compute_dist_matrix(self):
        # The pre-computed distance matrix must satisfy: D[i,j] = Loss( x(Tau[i,j]) - y(t[i]) )
        # Note: scipy.spatial.distance.cdist won't work since t is a vector and tau is a matrix.
        if self.verbose > 0: time_start = time.time()
        
        # We'll compute x(tau) and assign infinitity at undefined points.
        X = self.x_f(self.Tau)
        X[np.isnan(X)] = np.inf
        
        # We repeat y(t) so that it's the same shape of x(tau).
        Y = np.tile(self.y_f(self.t).reshape((self.N,1)),(1,self.M))
        
        # We apply the processed loss function.
        #   the default is "L2" => self.D = (X-Y)**2
        self.D = self.loss_f(X-Y) 
        # self.D = (X-Y)**2
        
        # Finally, we'll report the time it took to do all of this.
        if self.verbose > 0: print(f"Pre-computed loss: {time.time() - time_start :03.4f} sec")
        return self

    # --------------------------------------------------------------------------------------------
    # Solver 

    def run(self):
        self.check_params()
        self.allocate()
        self.offer_suggestions()
        self.iterate()
        return self

    def solve(self):
        time_start = time.time()
        i = solve(
            self.t,
            self.Tau,
            self.D,
            self.R_cum,
            self.R_inst,
            np.double(self.lambda_cum), 
            np.double(self.lambda_inst),
            np.double(self.s_min), 
            np.double(self.s_max),
            self.BC_start_stop,
            self.verbose,
            self.tau,
            self.path,
            self.f_tau
        )
        if i == -1: raise ValueError("C++ code failed.")
        self.time_solve = time.time() - time_start
        return self
    
    def iterate(self):
        for self.iteration in np.arange(self.max_iters).astype(np.int):
            # compute graph and solve
            self.compute_taus()
            self.compute_dist_matrix()
            self.solve()
            # optional methods
            if self.verbose > 1: self.print_iteration()
            if self.callback:    self.callback(self)
            # early termination
            if self.iteration > 0 and self.f_tau_ != np.inf:
                if np.abs(self.f_tau-self.f_tau_) <= self.epsilon_abs+self.epsilon_rel*np.abs(self.f_tau_):
                    if self.verbose > 2: print("Stopping criterion met.")
                    break
            self.f_tau_ = self.f_tau.copy()
        return self

    def print_iteration(self):
        if self.iteration==0: print(f'\titeration{" "*4}solver{" "*4}f(phi)')
        print(f'\t{self.iteration:3}/{self.max_iters:3}{" "*6}{self.time_solve:03.4f}{" "*3}{self.f_tau:8.6f}')
        return self

    def serialize(self):
        result = {
            "t"      : self.t,
            "tau"    : self.tau,
            "phi"    : self.phi,
            "y"      : self.y_a,
            "x"      : self.x_a,
            "x_hat"  : self.x_f(self.tau),
            "f_tau"  : self.f_tau.copy(),
            "params" : self.param_list
        }
        return result

    # --------------------------------------------------------------------------------------------
    # Helper methods for checking inputs

    def set_params(self, params={}):
        self.param_list = {k:v for k,v in params.items() if k not in ["x", "y"]}
        self.__dict__.update(params)
        return self

    def check_params(self):
        # If time is given as a sequence,
        if isinstance(self.t, np.ndarray) or isinstance(self.t, list):
            # we'll ensure it's a numpy array
            self.t = np.array(self.t, dtype=np.double)
            # and then check if it's multidimensional. 
            if self.t.ndim > 1:
                # If so we'll throw an error.
                raise ValueError(f"Time is multi-dimensional; we can only accept a 1-D sequence.")
            
            # If we're not given an N, we'll use the length of time as our N
            if not isinstance(self.N, int): # works for both int and np.int
                self.N = self.t.shape[0]
                # and alert the user.
                if self.verbose > 1:
                    print(f"Setting N={self.N} == len(t).")
            
            # Otherwise, we'll check to see if our given N and t agree.
            else:
                # If they agree, that's great,
                if self.N == self.t.shape[0]:
                    pass
                # otherwise, we'll need to choose one.
                else:
                    # If t is irregularly sampled (10 decimal place precision),
                    if np.unique(np.around(np.diff(self.t),10)).shape[0] > 1:
                        # we'll want to use that t to integrate over,
                        self.N = self.t.shape[0]
                        # and alert the user of this choice.
                        if self.verbose > 1:
                            print(f"Over-riding your choice of N: N = {self.N} == len(t). Since t is irregularly sampled, we'll want to integrate over that vector.")
                    # If N is bigger than t,
                    elif self.N > self.t.shape[0]:
                        # we'll use the smaller value,
                        self.N = self.t.shape[0]
                        # and alert the user of this choice.
                        if self.verbose > 1:
                            print(f"Over-riding your choice of N: N = {self.N} == len(t). Since N is greater than the length of t.")
                    # Otherwise,
                    else:
                        # we'll default to the value given by N and rebuild t, 
                        self.t = np.linspace(0, 1, num=self.N, dtype=np.double)
                        # and alert the user of this choice.
                        if self.verbose > 1:
                            print(f"You've set both t and N, but they don't agree: i.e. len(t) > N. We're keeping your choice of N = {self.N}.")

        # Otherwise, if N is given and t is not, we'll construct a sequence based on N.
        elif isinstance(self.N, int):
            self.t = np.linspace(0, 1, num=self.N, dtype=np.double)
        
        # It could be the case that both t and N are not given,
        else:
            # which will be indicated here,
            if self.t is None and self.N is None:
                if isinstance(self.x, np.ndarray) or isinstance(self.x, list):
                    self.N = np.array(self.x).shape[0]
                elif isinstance(self.y, np.ndarray) or isinstance(self.y, list):
                    self.N = np.array(self.y).shape[0]
                else:
                    # We'll use the default value of N
                    self.N = self.N_default
                self.t = np.linspace(0, 1, num=self.N, dtype=np.double)
                # and alert the user.
                if self.verbose > 1:
                    print(f"Setting N={self.N} since neither vector t or integer N is set.")
            # If we end up here, then the only explanation is that t is incorrect.
            else:
                raise ValueError(f"Time t is set incorrectly. It must be a 1-D sequence.")

        # If M is not given or is unreasonably large,
        if not isinstance(self.M, int) or self.M >= self.N:
            # we'll set it to the smaller of either the default size or a little over half of N,
            self.M = np.min((self.N*.55, self.M_max)).astype(np.int)
            # and alert the user of this choice.
            if self.verbose > 1:
                print(f"Setting M={self.M}")

        # We'll also ensure M is odd so that there's a center point (aka. j_center in our C++ code).
        self.M = self.M if self.M % 2 == 1 else self.M + 1

        # We'll alert the user of these final parameters.
        if self.verbose > 0:
            print(f"M={self.M}, N={self.N}")

        # Signals are given as generic inputs x and y. We'll parse these here.
        self.x_a, self.x_f = signal(self.x, "x", self.N)
        self.y_a, self.y_f = signal(self.y, "y", self.N)

        # Finally, we'll process our loss function.
        self.loss_f = process_function(self.Loss)

        return self

    def offer_suggestions(self):
        if self.verbose > 0:
            M_suggested = np.min((self.N*.55, self.M_max))
            
            if self.M < M_suggested:
                print(f"Suggestion: M is too small. Increasing M from {self.M} to ~{M_suggested} may offer better results.")

            if self.M > (self.N):
                print(f"Suggestion: M is too big. Decreasing M from {self.M} to ~{M_suggested} will be faster.")

            if (self.s_beta != 0 and (not callable(self.x) or not callable(self.y))):
                print(f"Suggestion:\n   x(t) and y(t) are defined over time domain [{0},{1}], but since you've set beta={self.s_beta}, this method will search over tau with a range of [{0-self.s_beta},{1+self.s_beta}].\n   The problem is that you've provided an array instead of a function for x(t) or y(t).\n   This method doesn't perform prediction, and so it won't impute values for x(tau) or y(tau) where tau < min(t) or tau > max(t).\n   Please make sure you use a function to define x and y instead of an array, or you'll have some spurrious results for tau outside the range of t.")
        
            range_x = [np.round(np.nanmin(self.x_f(self.t)),1), np.round(np.nanmax(self.x_f(self.t)),1)]
            range_y = [np.round(np.nanmin(self.y_f(self.t)),1), np.round(np.nanmax(self.y_f(self.t)),1)]
            
            if (range_x[0] != -1.0 or range_x[1] != 1.0 or range_y[0] != -1.0 or range_y[1] != 1.0):
                print(f"Suggestion: x(t) and y(t) do not have a range [-1,1] (they have range(x)={range_x} and range(y)={range_y}.")
                print(f"You may want to set scale_signals=True so the Loss function doesn't dominate the regularizers in the objective function.")
            
        return self


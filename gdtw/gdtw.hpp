/*
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Copyright (C) 2019-2024 Dave Deriso <dderiso@alumni.stanford.edu>, Twitter: @davederiso
 * Copyright (C) 2019-2024 Stephen Boyd
 * 
 * GDTW is a library that performs dynamic time warping.
 * GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization, 
 * which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters. 
 * 
 * Paper: https://rdcu.be/cT5dD
 * Source: https://github.com/dderiso/gdtw
 * Docs: https://dderiso.github.io/gdtw
 */

#pragma once

#include <vector>
#include <numeric>
#include <cmath>
#include <iterator>
#include <iostream>
#include <cfloat>
#include <Python.h>
#include <functional>

#define INFINITY 0x7f800000UL // std::numeric_limits<double>::infinity()

// loss functionals
#define L1_PENALTY(x) std::abs(x)
#define L2_PENALTY(x) (x)*(x)

// boundary conditions
#define DOUBLE_PRECISION_EPSILON 1e-10 // source of subtle errors -- really test this before changing
#define OUT_OF_BOUNDS(x, lower, upper) ((x < (lower - DOUBLE_PRECISION_EPSILON)) || (x > (upper + DOUBLE_PRECISION_EPSILON)))

int solve(
    // inputs
    const int &N, 
    const int &M,
    double* &t, 
    double* &Tau, 
    double* &D, 

    // parameters
    std::function<double(const double&)> &R_cuml, 
    std::function<double(const double&)> &R_inst, 
    double &lambda_cuml, 
    double &lambda_inst, 
    double &s_min, 
    double &s_max, 
    bool &BC_start_stop,

    // outputs
    double* &tau,
    int* &path, 
    double &f_of_tau
){
    // solver state space
    double* f = new double[N * M];
    double* n = new double[N * M];
       int* p = new    int[N * M];
    
    // for readability
    #define   D(i,j)   D[(i)*M + (j)]
    #define Tau(i,j) Tau[(i)*M + (j)]
    #define   f(i,j)   f[(i)*M + (j)]
    #define   n(i,j)   n[(i)*M + (j)]
    #define   p(i,j)   p[(i)*M + (j)]

    // fill node costs and init f(i,j) = inf
    int i,j,k;
    for (i=0; i<N; i++){
        for (j=0; j<M; j++){
            n(i,j) = D(i,j) + lambda_cuml * R_cuml( Tau(i,j) - t[i] ); 
            f(i,j) = INFINITY;
        }
    }

    // init by filling i=0
    const int j_center = (M-1)/2 + 1; // M is always odd
    for (j=0; j<M; j++){
        f(0,j) = n(0,j);
        if (BC_start_stop && j != j_center){
            f(0,j) = INFINITY; // enforce t_0 = 0
        }
    }

    // fill i=1 ... N-1
    double delta_t, slope, e_ijk, total_;
    for (i=0; i<N-1; i++){ // i=0 ... N-2 ( f(i+1,k) is filled )
        delta_t = t[i+1] - t[i];
        for (j=0; j<M; j++){
            for (k=0; k<M; k++){
                slope  = ( Tau(i+1,k) - Tau(i,j) ) / delta_t;
                if (OUT_OF_BOUNDS(slope, s_min, s_max)) continue; // boundary conditions
                e_ijk  = lambda_inst * R_inst(slope); // edge cost
                total_ = f(i,j) + delta_t * ( e_ijk + n(i+1,k) ); // Bellman
                if (total_ < f(i+1,k)){
                    f(i+1,k) = total_; // min
                    p(i+1,k) = j; // argmin
                }
            }
        }
    }

    // find terminal point of path
    int j_opt;
    double min,min_;
    if(BC_start_stop){
        j_opt = j_center; // enforce t_N = 1
    } 
    else {
        // argmin (unordered, linear search)
        min = INFINITY;
        for (j=0; j<M; j++){
            min_ = f(N-1,j);
            if (min_ < min ){
                min  = f(N-1,j);
                j_opt = j;
            }
        }
    }

    // net cost
    f_of_tau = f(N-1,j_opt);

    // re-trace path from terminal to origin point
    for (i=N-1; i>-1; i--){ // i = 0 ... N-1
        tau[i]  = Tau(i,j_opt);
        path[i] = j_opt;
        j_opt   = p(i,j_opt);
    }

    // free memory
    delete[] f;
    delete[] n;
    delete[] p;

    return 1;
}
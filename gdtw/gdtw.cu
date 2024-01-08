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
#define R_CUML(x) L2_PENALTY(x)
#define R_INST(x) L2_PENALTY(x)

// boundary conditions
#define DOUBLE_PRECISION_EPSILON 1e-10 // source of subtle errors -- really test this before changing
#define OUT_OF_BOUNDS(x, lower, upper) ((x < (lower - DOUBLE_PRECISION_EPSILON)) || (x > (upper + DOUBLE_PRECISION_EPSILON)))

// readability
#define       D(i,j)   D[(i)*M + (j)]
#define     Tau(i,j) Tau[(i)*M + (j)]
#define       f(i,j)   f[(i)*M + (j)]
#define       n(i,j)   n[(i)*M + (j)]
#define       p(i,j)   p[(i)*M + (j)]
#define   total(j,k)   p[(j)*M + (k)]


__global__ void kernel(
    // output
    double* total,

    // state
    const int i, 
    const double dt,  
    double* f, 
    double* n,

    // problem 
    const double s_min, 
    const double s_max,
    double lambda_inst,
    const int M, 
    const double* Tau
){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    double slope = ( Tau(i+1,k) - Tau(i,j) ) / dt;
    if (OUT_OF_BOUNDS(slope, s_min, s_max)) {
        total(j,k) = INFINITY;
    } else {
        // Bellman
        total(j,k) = f(i,j) + dt * ( 
            n(i+1,k) + // node cost
            ( lambda_inst * R_INST(slope) ) // edge cost
        );
    }
}


int solve_cuda(
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
    double* f;
    double* n;
    int* p;
    double* total;

    // CUDA setup
    cudaMallocManaged(&f, N * M * sizeof(double));
    cudaMallocManaged(&n, N * M * sizeof(double));
    cudaMallocManaged(&p, N * M * sizeof(int));
    cudaMallocManaged(&total, N * M * sizeof(double));
    dim3 threadsPerBlock(M, M);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // fill node costs and init f(i,j) = inf
    int i,j,k;
    double dt;
    for (i=0; i<N; i++){
        for (j=0; j<M; j++){
            dt = Tau(i,j) - t[i];
            n(i,j) = D(i,j) + lambda_cuml * R_CUML(dt); 
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
    for (i=0; i<N-1; i++){ // i=0 ... N-2 ( f(i+1,k) is filled )
        dt = t[i+1] - t[i];
        
        // distribute over j,k (parallelizes M*M operations)
        kernel<<<numBlocks, threadsPerBlock>>>(
            // output
            total,
            // state
            i, dt, f, n,
            // problem 
            s_min, s_max, lambda_inst, M, Tau
        );
        cudaDeviceSynchronize();

        // find optimum at i (unordered, linear search)
        for (j=0; j<M; j++){
            for (k=0; k<M; k++){
                if (total(j,k) < f(i+1,k)){
                    f(i+1,k) = total(j,k); // min
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
    *f_of_tau = f(N-1,j_opt);

    // re-trace path from terminal to origin point
    for (i=N-1; i>-1; i--){ // i = 0 ... N-1
        tau[i]  = Tau(i,j_opt);
        path[i] = j_opt;
        j_opt   = p(i,j_opt);
    }

    // free memory
    cudaFree(f);
    cudaFree(n);
    cudaFree(p);
    cudaFree(total);
}

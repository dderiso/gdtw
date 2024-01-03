/*
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Copyright (C) 2019-2024 Dave Deriso <dderiso@alumni.stanford.edu>, Twitter: @davederiso
 * Copyright (C) 2019-2024 Stephen Boyd
 * 
 * GDTW is a Python/C++ library that performs dynamic time warping.
 * GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization, 
 * which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters. 
 * 
 * Paper: https://rdcu.be/cT5dD
 * Source: https://github.com/dderiso/gdtw
 * Docs: https://dderiso.github.io/gdtw
 */


/*
clang -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /Users/dderiso/anaconda3/include -arch arm64 -fPIC -O2 -isystem /Users/dderiso/anaconda3/include -arch arm64 -I/Users/dderiso/anaconda3/lib/python3.11/site-packages/numpy/core/include -I/Users/dderiso/anaconda3/include/python3.11 -c gdtw/gdtw_solver.cpp -o build/temp.macosx-11.1-arm64-cpython-311/gdtw/gdtw_solver.o -Ofast -Wall
*/

#pragma once

#include <vector>
#include <numeric>
#include <cmath>
#include <iterator>
#include <iostream>
#include <cfloat>
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>

// wrapper for Numpy Arrays
#include "numpyobject.hpp"


#define DOUBLE_PRECISION_EPSILON 1e-10 // source of subtle errors -- really test this before changing
#define OUT_OF_BOUNDS(x, lower, upper) ((x < (lower - DOUBLE_PRECISION_EPSILON)) || (x > (upper + DOUBLE_PRECISION_EPSILON)))


class GDTW {
private:
    std::function<double(const double&, const double&, const double&)> R_inst; // instantaneous penalty
    std::function<double(const double&)> R_cum;  // cumulative penalty

public:

    GDTW& set_input_instantaneous_loss_functional(PyObject*& obj) {
        if(PyCallable_Check(obj)){
            R_inst = [obj](const double& x, const double& smin, const double& smax) -> double { 
                if(OUT_OF_BOUNDS(x, smin, smax)) return DBL_MAX;  // NPY_INFINITY
                return PyFloat_AsDouble(PyObject_CallFunction(obj,"f",x)); 
            };
            return *this;
        }

        if(!PyObject_TypeCheck(obj, &PyUnicode_Type)){
            throw std::runtime_error("set_input_instantaneous_loss_functional: Unhandled type for NumpyObject: " + std::string(Py_TYPE(obj)->tp_name) + ". Please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
        }

        if(PyUnicode_CompareWithASCIIString(obj,"L1") == 0){ 
            R_inst = [](const double& x, const double& smin, const double& smax) -> double {
                if(OUT_OF_BOUNDS(x, smin, smax)) return DBL_MAX;  // NPY_INFINITY
                return std::abs(x); 
            };
            return *this;
        }

        if(PyUnicode_CompareWithASCIIString(obj,"L2") == 0){ 
            R_inst = [](const double& x, const double& smin, const double& smax) -> double {
                if(OUT_OF_BOUNDS(x, smin, smax)) return DBL_MAX;  // NPY_INFINITY
                return x*x; 
            };
            return *this;
        }

        throw std::runtime_error("set_input_instantaneous_loss_functional: Unknown string: " + std::string(PyUnicode_AsUTF8(obj)) + ". Acceptable strings are either 'L1' or 'L2'. If you feel this error is incorrect, please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
    }

    GDTW& set_input_cumulative_loss_functional(PyObject*& obj) {
        if(PyCallable_Check(obj)){
            R_cum = [obj](const double& x) -> double { return PyFloat_AsDouble(PyObject_CallFunction(obj,"f",x)); };
            return *this;
        }

        if(!PyObject_TypeCheck(obj, &PyUnicode_Type)){
            throw std::runtime_error("set_input_cumulative_loss_functional: Unhandled type for NumpyObject: " + std::string(Py_TYPE(obj)->tp_name) + ". Please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
        }

        if(PyUnicode_CompareWithASCIIString(obj,"L1") == 0){ 
            R_cum = [](const double& x) -> double { return std::abs(x); };
            return *this;
        }

        if(PyUnicode_CompareWithASCIIString(obj,"L2") == 0){ 
            R_cum = [](const double& x) -> double { return x*x; };
            return *this;
        }

        throw std::runtime_error("set_input_cumulative_loss_functional: Unknown string: " + std::string(PyUnicode_AsUTF8(obj)) + ". Acceptable strings are either 'L1' or 'L2'. If you feel this error is incorrect, please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
    }

private:
    int verbosity;

    // inputs
    NumpyObject D;
    NumpyObject Tau;
    NumpyObject t;

    // outputs
    double* tau;
    int* path;
    double* score; // final score is given as f(tau)

    // methods
    

public:
    GDTW(int verbosity) : verbosity(verbosity) {}
    
    // double* D_buffer;
    // uint64_t D_stride_i, D_stride_j;
    GDTW& set_input_distance_matrix(PyObject*& obj) {
        D = NumpyObject(obj, 'D', verbosity);
        // D_buffer = (double*)PyArray_BYTES((PyArrayObject*) obj);
		// uint64_t* strides = (uint64_t*) PyArray_STRIDES((PyArrayObject*) obj);
		// D_stride_i = strides[0]/8;
		// D_stride_j = strides[1]/8;
        // #define D(i,j) D_buffer[i*D_stride_i + j*D_stride_j]
        return *this;
    }

    // double* Tau_buffer;
    // uint64_t Tau_stride_i, Tau_stride_j;
    // long* Tau_shape;
    GDTW& set_input_tau_matrix(PyObject*& obj) {
        Tau = NumpyObject(obj, 'T', verbosity);

        // Tau_shape = PyArray_SHAPE((PyArrayObject*) obj);
        // Tau_buffer = (double*)PyArray_BYTES((PyArrayObject*) obj);
		// uint64_t* strides = (uint64_t*) PyArray_STRIDES((PyArrayObject*) obj);
		// Tau_stride_i = strides[0]/8;
		// Tau_stride_j = strides[1]/8;
        // #define Tau(i,j) Tau_buffer[i*Tau_stride_i + j*Tau_stride_j]
        return *this;
    }

    GDTW& set_input_time_vector(PyObject*& obj) {
        t = NumpyObject(obj, 't', verbosity);
        return *this;
    }

    GDTW& set_output_tau_matrix(PyArrayObject*& obj) {
        tau = (double*) PyArray_DATA(obj);
        return *this;
    }

    GDTW& set_output_path_vector(PyArrayObject*& obj) {
        path = (int*) PyArray_DATA(obj);
        return *this;
    }

    GDTW& set_output_score(PyFloatObject*& obj) {
        score = &PyFloat_AS_DOUBLE(obj); // final score is given as f(tau)
        return *this;
    }

    GDTW& solve(const double& lambda_cum, const double& lambda_inst, const double& s_min, const double& s_max, const bool& BC_start_stop) {
        // graph dimensions are based on Tau \in R^{NxM}.
        const int N          = (int) Tau.shape[0];
        const int M          = (int) Tau.shape[1];
        const int j_center   = (M-1)/2 + 1; // M is always odd
        
        // solver state space
        // PyArrayObject* f = (PyArrayObject*)PyArray_SimpleNew(2, Tau.shape, NPY_DOUBLE);
        PyArrayObject* n = (PyArrayObject*)PyArray_SimpleNew(2, Tau.shape, NPY_DOUBLE);
        PyArrayObject* p = (PyArrayObject*)PyArray_SimpleNew(2, Tau.shape, NPY_INT);

        // // readability: f,n,p are 2D arrays and we don't need the overhead of NumpyObject
        // #define f(i,j) *(double*)PyArray_GETPTR2(f,i,j)
        #define n(i,j) *(double*)PyArray_GETPTR2(n,i,j)
        #define p(i,j)    *(int*)PyArray_GETPTR2(p,i,j)

        double* f = (double*) malloc(N*M*sizeof(double));
        // memset(f, 0, N*M*sizeof(double));
        #define f(i,j) f[i*N + j]

        // double* n = (double*) malloc(N*M*sizeof(double));
        // memset(n, 0, N*M*sizeof(double));
        // #define n(i,j) n[i*N + j]

        // int* p = (int*) malloc(N*M*sizeof(int));
        // memset(p, 0, N*M*sizeof(int));
        // #define p(i,j) p[i*N + j]

        // fill node costs and init f(i,j) = inf
        int i,j,k;
        for (i=0; i<N; i++){
            for (j=0; j<M; j++){
                n(i,j) = D(i,j) + lambda_cum * R_cum( Tau(i,j) - t[i] ); 
                f(i,j) = NPY_INFINITY;
            }
        }

        // init by filling i=0
        for (j=0; j<M; j++){
            f(0,j) = n(0,j);
            if (BC_start_stop && j != j_center){
                f(0,j) = NPY_INFINITY; // enforce t_0 = 0
            }
        }

        // fill i=1 ... N-1
        double delta_t, slope, e_ijk, total_;
        for (i=0; i<N-1; i++){ // i=0 ... N-2 ( f(i+1,k) is filled )
            delta_t = t[i+1] - t[i];
            for (j=0; j<M; j++){
                for (k=0; k<M; k++){
                    slope  = ( Tau(i+1,k) - Tau(i,j) ) / delta_t;
                    e_ijk  = lambda_inst * R_inst(slope, s_min, s_max); // edge cost
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
        } else {
            // argmin (unordered, linear search)
            min = NPY_INFINITY;
            for (j=0; j<M; j++){
                min_ = f(N-1,j);
                if (min_ < min ){
                    min  = f(N-1,j);
                    j_opt = j;
                }
            }
        }

        // net cost
        *score = f(N-1,j_opt);

        // re-trace path from terminal to origin point
        for (i=N-1; i>-1; i--){ // i = 0 ... N-1
            tau[i]  = Tau(i,j_opt);
            path[i] = j_opt;
            j_opt   = p(i,j_opt);
        }

        // free memory
        free(f);
        // free(n);
        // free(p);

        return *this;
    }

};
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

#include <vector>
#include <numeric>
#include <cmath>
#include <iterator>
#include <iostream>
#include <cfloat>
#include <Python.h>
#include <functional>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>

// wrapper for Numpy Arrays
#include "numpyobject.hpp"

// boundary checks and norms
#define DOUBLE_PRECISION_EPSILON 1e-10 // source of subtle errors -- really test this before changing
#define OUT_OF_BOUNDS(x, lower, upper) ((x < (lower - DOUBLE_PRECISION_EPSILON)) || (x > (upper + DOUBLE_PRECISION_EPSILON)))

// get type of object (function or string)
int get_function_type(PyObject*& obj){
    // Python function
    if(PyCallable_Check(obj)) return 2;

    // C++ Function
    if(!PyObject_TypeCheck(obj, &PyUnicode_Type)) throw std::runtime_error("get_function_type: Unhandled type for NumpyObject: " + std::string(Py_TYPE(obj)->tp_name) + ". Please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
    if(PyUnicode_CompareWithASCIIString(obj,"L2") == 0) return 0;
    if(PyUnicode_CompareWithASCIIString(obj,"L1") == 0) return 1;
    throw std::runtime_error("set_input_instantaneous_loss_functional: Unknown string: " + std::string(PyUnicode_AsUTF8(obj)) + ". Acceptable strings are either 'L1' or 'L2'. If you feel this error is incorrect, please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
}

static PyObject* gdtwcpp_solve(PyObject *self, PyObject *args){
    // shared pointers with Python
    PyObject *R_cuml_obj, *R_inst_obj, *t_obj, *Tau_obj, *D_obj;
    PyArrayObject *tau_obj, *path_obj;
    PyFloatObject *f_of_tau_obj;

    // values obtained from Python
    double lambda_cum, lambda_inst, s_min, s_max;
    bool BC_start_stop;
    int  verbose;

    // arg parse
    if (!PyArg_ParseTuple(args, "OOOOOddddpiO!O!O!",
        &t_obj,
        &Tau_obj,
        &D_obj,
        &R_cuml_obj, 
        &R_inst_obj,
        &lambda_cum,
        &lambda_inst,
        &s_min,
        &s_max,
        &BC_start_stop,
        &verbose,
        &PyArray_Type, &tau_obj,  // output
        &PyArray_Type, &path_obj, // output
        &PyFloat_Type, &f_of_tau_obj  // output
    )) return NULL;

    // loss functionals
    std::function<double(const double&)> R_cuml;
    std::function<double(const double&)> R_inst;
    const int r_cuml_type = get_function_type(R_cuml_obj);
    const int r_inst_type = get_function_type(R_inst_obj);
    if(r_cuml_type == 0) R_cuml = [](const double& x) { return x*x; }; // L2
    if(r_inst_type == 0) R_inst = [](const double& x) { return x*x; };
    if(r_cuml_type == 1) R_cuml = [](const double& x) { return std::abs(x); }; // L1
    if(r_inst_type == 1) R_inst = [](const double& x) { return std::abs(x); };
    if(r_cuml_type == 2) R_cuml = [R_cuml_obj](const double& x) { return PyFloat_AsDouble(PyObject_CallFunction(R_cuml_obj,"f",x)); }; // Python function
    if(r_inst_type == 2) R_inst = [R_inst_obj](const double& x) { return PyFloat_AsDouble(PyObject_CallFunction(R_inst_obj,"f",x)); };
    
    // inputs
    NumpyObject t   = NumpyObject(t_obj, 't', verbose);
    NumpyObject Tau = NumpyObject(Tau_obj, 'T', verbose);
    NumpyObject D   = NumpyObject(D_obj, 'D', verbose);

    // graph dimensions are based on Tau \in R^{NxM}.
    const int N          = (int) Tau.shape[0];
    const int M          = (int) Tau.shape[1];
    const int j_center   = (M-1)/2 + 1; // M is always odd

    // outputs
    double* tau      = (double*) PyArray_DATA(tau_obj);
    int*    path     =    (int*) PyArray_DATA(path_obj);
    double* f_of_tau = &PyFloat_AS_DOUBLE(f_of_tau_obj);

    // solver state space
    PyArrayObject* f = (PyArrayObject*)PyArray_SimpleNew(2, Tau.shape, NPY_DOUBLE);
    PyArrayObject* n = (PyArrayObject*)PyArray_SimpleNew(2, Tau.shape, NPY_DOUBLE);
    PyArrayObject* p = (PyArrayObject*)PyArray_SimpleNew(2, Tau.shape, NPY_INT);

    // readability: f,n,p are 2D arrays and we don't need the overhead of NumpyObject
    #define f(i,j) *(double*)PyArray_GETPTR2(f,i,j)
    #define n(i,j) *(double*)PyArray_GETPTR2(n,i,j)
    #define p(i,j)    *(int*)PyArray_GETPTR2(p,i,j)

    // fill node costs and init f(i,j) = inf
    int i,j,k;
    for (i=0; i<N; i++){
        for (j=0; j<M; j++){
            n(i,j) = D(i,j) + lambda_cum * R_cuml( Tau(i,j) - t[i] ); 
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
                e_ijk  = lambda_inst * (OUT_OF_BOUNDS(slope, s_min, s_max) ? DBL_MAX : R_inst(slope)); // edge cost, use NPY_INFINITY instead of DBL_MAX ?
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
    *f_of_tau = f(N-1,j_opt);

    // re-trace path from terminal to origin point
    for (i=N-1; i>-1; i--){ // i = 0 ... N-1
        tau[i]  = Tau(i,j_opt);
        path[i] = j_opt;
        j_opt   = p(i,j_opt);
    }

    // free memory
    Py_DECREF(f);
    Py_DECREF(n);
    Py_DECREF(p);

    return Py_BuildValue("i", 1);
}

static PyObject *gdtwcpp_test(PyObject *self, PyObject *args){
    return Py_BuildValue("i", 1);
}

static PyMethodDef gdtwcpp_methods[] = {
    {"solve",   gdtwcpp_solve,  METH_VARARGS, "Runs the Viterbi algorithm and graph computation together."},
    {"test",    gdtwcpp_test,   METH_VARARGS, "Tests that the library loaded."},
    {NULL,      NULL}       /* sentinel */
};

static struct PyModuleDef gdtwcpp_module = {
    PyModuleDef_HEAD_INIT,
    "gdtwcpp",   /* name of module */
    "General Dynamic Time Warping, C++ Library", /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    gdtwcpp_methods
};

PyMODINIT_FUNC PyInit_gdtwcpp(void){
    import_array();
    return PyModule_Create(&gdtwcpp_module);
}

int main(int argc, char *argv[]){
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    PyImport_AppendInittab("gdtwcpp", PyInit_gdtwcpp);
    // Py_SetProgramName is deprecated
    // Py_SetProgramName(program);
    Py_Initialize();
    PyImport_ImportModule("gdtwcpp");
    PyMem_RawFree(program);
    return 0;
}

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

#include <iostream>
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>

#include "gdtw.hpp"

// get type of object (function or string)
void set_loss_functional(PyObject*& obj, std::function<double(const double&)>& func){
    // Python function
    if(PyCallable_Check(obj)) {
        func = [obj](const double& x) { return PyFloat_AsDouble(PyObject_CallFunction(obj,"f",x)); };
        return;
    }

    // C++ Function (indexed by string)
    if(!PyObject_TypeCheck(obj, &PyUnicode_Type)) throw std::runtime_error("set_loss_functional: Unhandled type for NumpyObject: " + std::string(Py_TYPE(obj)->tp_name) + ". Please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
    if(PyUnicode_CompareWithASCIIString(obj,"L2") == 0) {
        func = [](const double& x) { return L2_PENALTY(x); };
        return;
    }
    if(PyUnicode_CompareWithASCIIString(obj,"L1") == 0) {
        func = [](const double& x) { return L1_PENALTY(x); };
        return;
    }
    throw std::runtime_error("set_loss_functional: Unknown string: " + std::string(PyUnicode_AsUTF8(obj)) + ". Acceptable strings are either 'L1' or 'L2'. If you feel this error is incorrect, please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
}

static PyObject* extract_python_variables_and_solve(PyObject *self, PyObject *args){
    // shared pointers with Python
    PyObject *R_cuml_obj, *R_inst_obj, *t_obj, *Tau_obj, *D_obj;
    PyArrayObject *tau_obj, *path_obj;
    PyFloatObject *f_of_tau_obj;

    // const values obtained from Python
    double lambda_cuml, lambda_inst, s_min, s_max;
    bool BC_start_stop;
    int  verbosity;

    // arg parse
    if (!PyArg_ParseTuple(args, "OOOOOddddpiO!O!O!",
        &t_obj, // time series t
        &Tau_obj, // time series Tau
        &D_obj, // time series D
        &R_cuml_obj, // cumulative loss function
        &R_inst_obj, // instantaneous loss function
        &lambda_cuml, // cumulative loss weight
        &lambda_inst, // instantaneous loss weight
        &s_min, // minimum slope
        &s_max, // maximum slope
        &BC_start_stop, // boundary condition flag
        &verbosity, // verbosity level
        &PyArray_Type, &tau_obj,  // output: warped time series
        &PyArray_Type, &path_obj, // output: optimal path
        &PyFloat_Type, &f_of_tau_obj  // output: optimal cost
    )) return NULL;

    // loss functionals
    std::function<double(const double&)> R_cuml;
    std::function<double(const double&)> R_inst;
    set_loss_functional(R_cuml_obj, R_cuml);
    set_loss_functional(R_inst_obj, R_inst);

    // inputs
    double* t   = (double*) PyArray_BYTES((PyArrayObject*) t_obj);
    double* D   = (double*) PyArray_BYTES((PyArrayObject*) D_obj);
    double* Tau = (double*) PyArray_BYTES((PyArrayObject*) Tau_obj);

    // outputs
    double* tau      = (double*) PyArray_BYTES(tau_obj);
    int*    path     =    (int*) PyArray_BYTES(path_obj);
    double& f_of_tau = f_of_tau_obj->ob_fval;

    // graph dimensions are based on Tau \in R^{NxM}.
    npy_intp* Tau_shape = ((PyArrayObject_fields *) Tau_obj)->dimensions;
    const int N          = (int) Tau_shape[0];
    const int M          = (int) Tau_shape[1];

    // run solver
    solve(N, M, t, Tau, D, R_cuml, R_inst, lambda_cuml, lambda_inst, s_min, s_max, BC_start_stop, tau, path, f_of_tau);

    return Py_BuildValue("i", 1);
}

static PyObject* test(PyObject *self, PyObject *args){
    return Py_BuildValue("i", 1);
}

static PyMethodDef methods[] = {
    {"solve",   extract_python_variables_and_solve,  METH_VARARGS, "Extracts Python variables and runs the solver."},
    {"test",    test,   METH_VARARGS, "Tests that the library loaded."},
    {NULL,      NULL}       /* sentinel */
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "gdtwcpp",   /* name of module */
    "General Dynamic Time Warping, C++ Library", /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods
};

PyMODINIT_FUNC PyInit_gdtwcpp(void){
    import_array();
    return PyModule_Create(&module);
}

int main(int argc, char *argv[]){
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    PyImport_AppendInittab("gdtwcpp", PyInit_gdtwcpp);
    Py_Initialize();
    PyImport_ImportModule("gdtwcpp");
    PyMem_RawFree(program);
    return 0;
}
/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2019-2026 Dave Deriso <dderiso@alumni.stanford.edu>, Twitter: @davederiso
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

// Built-in penalty strings dispatch as small codes (inlined switch in the
// kernel, GIL released). Returns false for non-string objects; throws for
// unknown strings, matching set_loss_functional's message.
static bool penalty_code_of(PyObject* obj, double huber_delta, GDTWPenaltyCode* out){
    if (!PyObject_TypeCheck(obj, &PyUnicode_Type)) return false;
    out->delta = huber_delta;
    if (PyUnicode_CompareWithASCIIString(obj, "L2") == 0)    { out->type = GDTW_PEN_L2;    return true; }
    if (PyUnicode_CompareWithASCIIString(obj, "L1") == 0)    { out->type = GDTW_PEN_L1;    return true; }
    if (PyUnicode_CompareWithASCIIString(obj, "huber") == 0) { out->type = GDTW_PEN_HUBER; return true; }
    throw std::runtime_error("set_loss_functional: Unknown string: " + std::string(PyUnicode_AsUTF8(obj)) + ". Acceptable strings are 'L1', 'L2', or 'huber'. If you feel this error is incorrect, please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
}

// get type of object (function or string)
void set_loss_functional(PyObject*& obj, std::function<double(const double&)>& func, double huber_delta){
    // Python function. "d" passes the argument at double precision (the "f"
    // format truncated to single precision), and the result must be released:
    // PyObject_CallFunction returns a new reference, which previously leaked
    // once per evaluation across the O(N*M^2) solver loop.
    if(PyCallable_Check(obj)) {
        func = [obj](const double& x) {
            PyObject* result = PyObject_CallFunction(obj, "d", x);
            if (result == NULL) throw std::runtime_error("set_loss_functional: user-supplied penalty raised an exception.");
            const double value = PyFloat_AsDouble(result);
            Py_DECREF(result);
            return value;
        };
        return;
    }

    // C++ Function (indexed by string)
    if(!PyObject_TypeCheck(obj, &PyUnicode_Type)) throw std::runtime_error("set_loss_functional: Unhandled type for NumpyObject: " + std::string(Py_TYPE(obj)->tp_name) + ". Please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
    GDTWPenaltyCode code;
    penalty_code_of(obj, huber_delta, &code); // throws on unknown strings
    func = [code](const double& x) { return code(x); };
}

static PyObject* extract_python_variables_and_solve(PyObject *self, PyObject *args){
    // shared pointers with Python
    PyObject *R_cuml_obj, *R_inst_obj, *t_obj, *Tau_obj, *D_obj;
    PyArrayObject *tau_obj, *path_obj;
    PyFloatObject *f_of_tau_obj;

    // const values obtained from Python
    double lambda_cuml, lambda_inst, s_min, s_max, huber_delta;
    bool BC_start_stop;
    int  verbosity;

    // arg parse
    if (!PyArg_ParseTuple(args, "OOOOOdddddpiO!O!O!",
        &t_obj, // time series t
        &Tau_obj, // time series Tau
        &D_obj, // time series D
        &R_cuml_obj, // cumulative loss function
        &R_inst_obj, // instantaneous loss function
        &lambda_cuml, // cumulative loss weight
        &lambda_inst, // instantaneous loss weight
        &s_min, // minimum slope
        &s_max, // maximum slope
        &huber_delta, // huber transition point (used only when R_cuml/R_inst == "huber")
        &BC_start_stop, // boundary condition flag
        &verbosity, // verbosity level
        &PyArray_Type, &tau_obj,  // output: warped time series
        &PyArray_Type, &path_obj, // output: optimal path
        &PyFloat_Type, &f_of_tau_obj  // output: optimal cost
    )) return NULL;

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

    // run solver; convert C++ exceptions (unknown penalty strings, raising
    // Python callbacks) into Python errors instead of terminating the
    // interpreter. Callbacks may already have set a Python error; keep it.
    int status = 0;
    try {
        GDTWPenaltyCode code_cuml, code_inst;
        const bool fast = !PyCallable_Check(R_cuml_obj) && !PyCallable_Check(R_inst_obj)
                          && penalty_code_of(R_cuml_obj, huber_delta, &code_cuml)
                          && penalty_code_of(R_inst_obj, huber_delta, &code_inst);
        if (fast) {
            // built-in penalties: no Python calls inside, release the GIL so
            // thread pools parallelize concurrent solves on real cores
            Py_BEGIN_ALLOW_THREADS
            status = solve_impl(N, M, t, Tau, D, code_cuml, code_inst,
                                lambda_cuml, lambda_inst, s_min, s_max,
                                BC_start_stop, tau, path, f_of_tau);
            Py_END_ALLOW_THREADS
        } else {
            // user-supplied Python penalties call back into the interpreter,
            // so the GIL is held
            std::function<double(const double&)> R_cuml;
            std::function<double(const double&)> R_inst;
            set_loss_functional(R_cuml_obj, R_cuml, huber_delta);
            set_loss_functional(R_inst_obj, R_inst, huber_delta);
            status = solve(N, M, t, Tau, D, R_cuml, R_inst,
                           lambda_cuml, lambda_inst, s_min, s_max,
                           BC_start_stop, tau, path, f_of_tau);
        }
    } catch (const std::exception& e) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    if (status != 0) {
        PyErr_SetString(PyExc_ValueError,
            "gdtwcpp.solve: no feasible warp path under the relaxed boundaries (slope band too tight for the candidate grid)");
        return NULL;
    }

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

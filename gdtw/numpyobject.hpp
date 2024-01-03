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

#pragma once

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>
#include <iostream>

#define NUMPY_OBJECT_NONE       0
#define NUMPY_OBJECT_FUNCTION   1
#define NUMPY_OBJECT_ARRAY      2

class NumpyObject {
	
private:
	int verbosity;
	PyObject* obj;

public:
	int type;
	int ndims;
	long* shape;
	double* data;
	long* strides;
	char name;

	uint64_t stride_i;
	uint64_t stride_j;

	NumpyObject() : verbosity(0), obj(nullptr), type(NUMPY_OBJECT_NONE), name('\0') {}

	NumpyObject(PyObject* obj_, char name_, const int& verbosity_) : verbosity(verbosity_), obj(obj_), name(name_) {
		type = get_type();
		Py_INCREF(obj);
		if(type == NUMPY_OBJECT_ARRAY) get_array_attributes();
		if(verbosity > 2)   print();
	}

	~NumpyObject(){
		// Py_XDECREF(obj);
	}

	int get_type(){
		if(obj == Py_None)        return NUMPY_OBJECT_NONE;
		if(PyCallable_Check(obj)) return NUMPY_OBJECT_FUNCTION;
		if(PyArray_Check(obj))    return NUMPY_OBJECT_ARRAY;
		
		throw std::runtime_error("Unhandled type for NumpyObject: " + std::string(Py_TYPE(obj)->tp_name) + ". Please create a GitHub issue at https://github.com/dderiso/gdtw/issues with this message and the inputs you used when calling the gdtw solver.");
	}

	bool is_array() const {
		return type == NUMPY_OBJECT_ARRAY;
	}

	void get_array_attributes(){
		ndims   = PyArray_NDIM((PyArrayObject*) obj);
		shape   = PyArray_SHAPE((PyArrayObject*) obj);
		data    = (double*)PyArray_BYTES((PyArrayObject*) obj);
		uint64_t* strides = ( uint64_t*) PyArray_STRIDES((PyArrayObject*) obj);
		stride_i = strides[0]/8;
		stride_j = strides[1]/8;
	}

	void print() const {
		printf("Object is type %i \n", type);
		if(type == NUMPY_OBJECT_ARRAY){
			printf("Object is %i dimensional \n", ndims);
			for (int i=0; i < ndims; i++){
				printf("  Dimension %i has %i elements \n", i, (int)shape[i]);
			}
		}
	}

	double& operator[] (const int& i) {
		return data[i];
	}
	
	double& operator() (const int& i) {
		return data[i];
	}

	double& operator() (const int& i, const int& j) {
		// return *(double*)PyArray_GETPTR2((PyArrayObject*) obj,i,j);
		return data[i*stride_i + j*stride_j];
	}

	// double& operator() (const int& i, const int& j, const int& k) {
	// 	// return *(double*)PyArray_GETPTR3((PyArrayObject*) obj,i,j,k);
	// 	return *(data + i*stride_i + j*strides[1] + k*strides[2]);
	// }
};
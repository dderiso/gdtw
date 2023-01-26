/*
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Copyright (C) 2019-2023 Dave Deriso <dderiso@alumni.stanford.edu>
 * Copyright (C) 2019-2023 Stephen Boyd
 * 
 * GDTW is a Python/C++ library that performs dynamic time warping.
 * It is based on a paper by Dave Deriso and Stephen Boyd.
 * GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization, 
 * which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters. 
 * 
 * Visit: https://github.com/dderiso/gdtw (source)
 * Visit: https://dderiso.github.io/gdtw  (docs) 
 */

#ifndef NUMPYOBJECT_H
#define NUMPYOBJECT_H

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>
#include <iostream>

class NumpyObject{
	private:
		PyObject* obj;
		int NONE = -1;
		int FUNCTION = 0;
		int ARRAY = 1;
		int verbose;
	public:
		int type;
		int ndims;
		long* shape;
		double* data;
		long* strides;
		double err = (double)NULL;
		int error = 0;
		char name;

		NumpyObject();
		NumpyObject(PyObject* obj_, char name_, const int& verbose_);
		~NumpyObject();

		void print();

		int& get_type();
		void get_array_attributes();
		bool is_array();

		double& operator[] (const int& i) {
			return data[i];
		}
		double& operator() (const int& i) {
			return data[i];
		}
		double& operator() (const int& i, const int& j) {
			return *(double*)PyArray_GETPTR2((PyArrayObject*) obj,i,j);
			// return data + i*strides[0] + j*strides[1];
		}

		double& operator() (const int& i, const int& j, const int& k) {
			return *(double*)PyArray_GETPTR3((PyArrayObject*) obj,i,j,k);
			// return data + i*strides[0] + j*strides[1] + k*strides[2];
		}
};


#endif
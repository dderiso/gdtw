#include "numpyobject.hpp"

NumpyObject::NumpyObject(){
}

NumpyObject::NumpyObject(PyObject* obj_, char name_, const int& verbose_){
	obj     = obj_;
	name    = name_;
	verbose = verbose_;
	Py_INCREF(obj);
	type = get_type();
	if(type == ARRAY) get_array_attributes();
	if(verbose > 2)   print();
}

NumpyObject::~NumpyObject(){
}

int& NumpyObject::get_type(){
	if(obj == Py_None){
		return NONE;
	} else {
		return PyCallable_Check(obj) ? FUNCTION : ARRAY;
	}	
	//todo throw error for unhandled type
}

bool NumpyObject::is_array(){
	return type == ARRAY;
}

void NumpyObject::get_array_attributes(){
	ndims   = PyArray_NDIM((PyArrayObject*) obj);
	shape   = PyArray_SHAPE((PyArrayObject*) obj);
	data    = (double*)PyArray_DATA((PyArrayObject*) obj);
	strides = PyArray_STRIDES((PyArrayObject*) obj);
}

void NumpyObject::print(){
	printf("Object is type %i \n", type);
	if(type == ARRAY){
		printf("Object is %i dimensional \n", ndims);
		for (int i=0; i < ndims; i++){
			printf("  Dimension %i has %i elements \n", i, (int)shape[i]);
		}
	}
}


#include "utils.hpp"

// get type of object (function or string)
int f_type(PyObject*& obj){
    int type = -1; // no match
    
    // Python function
    if(PyCallable_Check(obj)){ type = 0; }

    // C++ Function
    else if(PyObject_TypeCheck(obj, &PyUnicode_Type)){
             if(PyUnicode_CompareWithASCIIString(obj,"L1") == 0){ type = 1; }
        else if(PyUnicode_CompareWithASCIIString(obj,"L2") == 0){ type = 2; }
        else { printf("Error: String \"%s\" not in this API. \n", PyUnicode_AsUTF8(obj)); }
    }
    return type;
}


// bound checks
// DBL_EPSILON ?
bool out_of_bounds(const double& x, const double& lower, const double& upper){
    return (x < (lower - 1e-10)) || (x > (upper + 1e-10));
}

// norms
double L1(const double& x){ return abs(x); }

double L2(const double& x){ return x*x; }

double L1(const double* x, const double* y){ 
    double delta, out = 0.0;
    int x_M = sizeof(x)/sizeof(double);
    for(int j=0; j<x_M; j++){
        delta = x[j] - y[j];
        out += abs(delta);
    }
    return out;
}

double L2(const double* x, const double* y){ 
    double delta, out = 0.0;
    int x_M = sizeof(x)/sizeof(double);
    for(int j=0; j<x_M; j++){
        delta = x[j] - y[j];
        out += delta*delta;
    }
    return out; //sqrt(out);
}






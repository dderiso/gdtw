import numpy as np

def scale(seq, range=[-1,1]):
    return (range[1]-range[0])*((seq-np.nanmin(seq))/np.nanmax(seq-np.nanmin(seq))) + range[0]

def process_function(f):
    # Our API is a first pass. We'll improve this later. For now, we take the choice of 
    # loss and regularization functionals as a string or a callable function.
    # We'll parse this API here and return a callable function or None.

    # default
    f_out = None

    # Determine if the function is given as a string (indicates that the user wants a C++ function),
    if isinstance(f, str):
        # if so convert the string into a function.
        if   f == "L1": f_out = lambda x_,axis=1: np.linalg.norm(x_,ord=1,axis=axis)
        elif f == "L2": f_out = lambda x_,axis=1: np.linalg.norm(x_,ord=2,axis=axis)**2
        else:
            raise ValueError("Error: String is not recognized by this API. You can use one of the built-in C function by passing a string such as 'L1' or 'L2'.")

    # Or, if it's a callable function (indicates that the user wrote this function),
    elif callable(f):
        # we'll test it ability to be vectorized by running it on a test array and seeing if the shape is preserved.
        if f(np.zeros(10)).shape == 10:
            try:
                # If it's vectorizable, make it so,
                f_out = np.vectorize(f)
            except:
                # otherwise, alert the user
                if verbose > 1:
                    print("Could not vectorize function, proceeding with a loop (slower).")
                
                # and run the function as a loop (slower).
                f_out = lambda x_: [f for x_i in x_]

        # We'll test this to be sure.
        if f_out(np.zeros(10)).shape != 10:
            raise ValueError("Error: If you define your own Python function (slower than C), it must accept and return a matrix of the same shape. Yours does not.")
    
    # Finally, alert the user if the function is neither.
    else: 
        raise ValueError("Error: Function is neither a String nor callable function.")

    return f_out



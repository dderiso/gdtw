import numpy as np
t = np.linspace(0,1,1000)
q = lambda t_: t_**2
x = lambda t_: np.sin(2*np.pi * 5 * t_)
y = lambda t_: x(q(t_))

import gdtw
try:
    phi, x_tau, f_tau, g = gdtw.warp(x(t), y(t))
    print("Test passed")
    print(f_tau)
except Exception as e:
    print("Test failed due to error:", e)

# x_tau = list(x_tau.astype(np.float32))

# print(x_tau)

# import pathlib
# path = pathlib.Path(__file__).parent.resolve()

# import pickle
# # with open(f"{path}/test.pkl", 'wb') as f:
# #   pickle.dump(x_tau, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open(f"{path}/test.pkl", 'rb') as f:
#   x_tau_old = pickle.load(f)

# for i,(xi,yi) in enumerate(zip(x_tau, x_tau_old)):
# 	if np.abs(xi-yi) > 1E-2:
# 		print(i, xi, yi)

# import os
# import pickle
# import numpy as np

# if os.path.exists('x_tau.pkl'):
#     with open('x_tau.pkl', 'rb') as f:
#         x_tau_old = pickle.load(f)
#     mse_error = np.mean((x_tau_old - x_tau)**2)
#     print(f"MSE error between the old and current solution: {mse_error}")


import time
from tabulate import tabulate

# Initialize an empty list to store the performance times
performance_times = []

# Define the sizes of t to test
t_sizes = [10, 100, 1000, 10000] #, 100000]

# Loop over the sizes of t
for t_size in t_sizes:
    print(f"Testing t = {t_size}")
    # Initialize a list to store the times for this size
    times_for_this_size = []
    # Repeat the test 3 times
    for _ in range(3):
        t_sub = np.linspace(0,1,t_size)
        start_time = time.time()
        phi, x_tau, f_tau, g = gdtw.warp(x(t_sub), y(t_sub))#, params={"Loss": loss})
        end_time = time.time()
        times_for_this_size.append(end_time - start_time)
    # Append the average time for this size to the performance times
    performance_times.append(sum(times_for_this_size) / len(times_for_this_size))
    print(f"avg = {performance_times[-1]}")

# Print the performance times in a table
print(tabulate(zip(t_sizes, performance_times), headers=['Size of t', 'Average Time']))
import numpy as np
t = np.linspace(0,1,1000)
q = lambda t_: t_**2
x = lambda t_: np.sin(2*np.pi * 5 * t_)
y = lambda t_: x(q(t_))

import gdtw
phi, x_tau, f_tau, g = gdtw.warp(x(t), y(t))

x_tau = list(x_tau.astype(np.float32))

import pathlib
path = pathlib.Path(__file__).parent.resolve()

import pickle
# with open(f"{path}/test.pkl", 'wb') as f:
#   pickle.dump(x_tau, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"{path}/test.pkl", 'rb') as f:
  x_tau_old = pickle.load(f)

for i,(xi,yi) in enumerate(zip(x_tau, x_tau_old)):
	if np.abs(xi-yi) > 1E-2:
		print(i, xi, yi)
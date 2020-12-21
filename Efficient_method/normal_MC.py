# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:52:33 2020

@author: Jing
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from SALib.sample import saltelli

# Global settings
seed = 2**30
plt.style.use('default')

Ma = 2
Mb = 2
Mc = 2

PMA = np.array([0.5, 0.5])
PMB = np.array([0.5, 0.5])
PMC = np.array([0.5, 0.5])

# Parameters for snow melt process
P = 60                # Precipation in inch/yr
Ta = 7                # Average temperature for a given day in degree 
Tm = 0                # Critical snow melt point in degree
Csn = 0.8             # Runoff confficient
SVC = 0.7             # Snow cover fraction 
A = 2000 * 1e6        # Upper catchment area in  km2
Rn = 80               # Surface radiation in w/m2

# Left boundary condition
h1 = 300              # Head in the left 

# Domain information
z0 = 289              # Elevation of river bed in meters    
L = 10000 
x0 = 7000
nx = 101
x = np.linspace(0, L, nx, endpoint=True)

# Parameter bounds and distributions
# Parameters for snow melt process
P = 60                # Precipation in inch/yr
Ta = 7                # Average temperature for a given day in degree 
Tm = 0                # Critical snow melt point in degree
Csn = 0.8             # Runoff confficient
SVC = 0.7             # Snow cover fraction 
A = 2000 * 1e6        # Upper catchment area in  km2
Rn = 80               # Surface radiation in w/m2

# Left boundary condition
h1 = 300              # Head in the left 

# Domain information
z0 = 289              # Elevation of river bed in meters    
L = 10000   
x0 = 7000
Nx = 21
qid = 14
X = np.linspace(0, L, Nx, endpoint=True)

# Parameter bounds and distributions
bounds = {'a' : [2.0, 0.4],
          'b' : [0.2, 0.5],
          'hk': [2.9, 0.5],
          'k1': [2.6, 0.3],
          'k2': [3.2, 0.3],
          'f1': [3.5, 0.75],
          'f2': [2.5, 0.3],
          'r' : [0.3, 0.05]}

dists = {'a' : 'norm',
         'b' : 'unif',
         'hk': 'lognorm',
         'k1': 'lognorm',
         'k2': 'lognorm',
         'f1': 'norm',
         'f2': 'norm',
         'r' : 'norm'}

problem = {'num_vars': 8,
           'names': ['a', 'b', 'hk', 'k1', 'k2', 'f1', 'f2', 'r'],
           'bounds': [bounds['a'], bounds['b'], bounds['hk'], bounds['k1'], bounds['k2'], bounds['f1'], bounds['f2'], bounds['r']],
           'dists': [dists['a'], dists['b'], dists['hk'], dists['k1'], dists['k2'], dists['f1'], dists['f2'], dists['r']]
           }

@nb.njit(parallel=True)
def cmpt_Dscs(param_values):
    N = param_values.shape[0]
    Y = np.zeros((Ma, N, Mb, N, Mc, N))
    
    for i in range(Ma):
        for j in nb.prange(N):
            if i == 0:
                a = param_values[j, 0]
                w = a * (P - 14)**0.5 * 25.4 * 0.001 / 365
            else:
                b = param_values[j, 1]
                w = b * (P - 15.7) * 25.4 * 0.001 / 365
                
            for k in range (Mb):
                for l in nb.prange(N):
                    if k == 0:
                        k1 = param_values[l, 2]
                        k2 = param_values[l, 2]
                    else:
                        k1 = param_values[l, 3]
                        k2 = param_values[l, 4]
                        
                    for m in range(Mc):
                        for n in nb.prange(N):
                            if m == 0:
                                f1 = param_values[n, 5]
                                M = f1 * (Ta - Tm)
                                Q = Csn * M * SVC * A * 0.001 / 86400
                                h2 = 0.3 * Q**0.6 + z0
                            else:
                                f2 = param_values[n, 6]
                                r = param_values[n, 7]
                                M = f2 * (Ta - Tm) + r * Rn
                                Q = Csn * M * SVC * A * 0.001 / 86400
                                h2 = 0.3 * Q**0.6 + z0

                            C1 = (h1**2 - h2**2 - w / k1 * x0**2 + w / k2 * x0**2 - w / k2 * L**2) / (k1 / k2 * x0 - k1 / k2 * L - x0)
                            Y[i, j, k, l, m, n] =  w * x0 - k1 * C1 / 2
                            
    return Y

@nb.njit(nogil=True)
def normal_MC(n):
    Y = Y_base[:, :n, :, :n, :, :n]
    N = Y.shape[1]
    Var_t_d = np.var(Y)
    
    E_tc_d = np.zeros((Ma, N, Mb, N, Mc))
    E_c_d = np.zeros((Ma, N, Mb, N))
    E_tb_d = np.zeros((Ma, N, Mb))
    E_tb_d2 = np.zeros((Ma, N, Mb))
    E_b_d = np.zeros((Ma, N))
    E_b_d2 = np.zeros((Ma, N))

    E_ta_d = np.zeros(Ma)
    E_ta_d2 = np.zeros(Ma)
    

    for i in range(Ma):
        for j in range(N):
            for k in range(Mb):
                for l in range(N):
                    for m in range(Mc):
                        E_tc_d[i, j, k, l, m] = np.mean(Y[i, j, k, l, m, :])
                    E_c_d[i, j, k, l] = PMC[0] * E_tc_d[i, j, k, l, 0] + PMC[1] * E_tc_d[i, j, k, l, 1]
                E_tb_d[i, j, k] = np.mean(E_c_d[i, j, k, :])
                E_tb_d2[i, j, k] = np.mean(E_c_d[i, j, k, :]**2)
            E_b_d[i, j] = PMB[0] * E_tb_d[i, j, 0] + PMB[1] * E_tb_d[i, j, 1]
            E_b_d2[i, j] = PMB[0] * E_tb_d2[i, j, 0] + PMB[1] * E_tb_d2[i, j, 1]
        E_ta_d[i] = np.mean(E_b_d[i, :])
        E_ta_d2[i] = np.mean(E_tb_d2[i, :])
    E_a_d = PMA[0] * E_ta_d[0] + PMA[1] * E_ta_d[1]
    E_a_d2 = PMA[0] * E_ta_d2[0] + PMA[1] * E_ta_d2[1]
        
    Var_C = E_a_d2 - E_a_d**2
    ST_C = 1 - Var_C / (Var_t_d + 1e-20)
    
    print('n=', n, 'ST_C=', ST_C)
    
    return ST_C


# Generate parameters using SALib.sample.saltelli
N = 500
param_values_base = saltelli.sample(problem, N, calc_second_order=False, seed=seed)[::10, :]

# Compute the output
Y_base = cmpt_Dscs(param_values_base)

# Compute the index
print(normal_MC(N))


from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=6) as executor:
    # executor.map(normal_MC, np.arange(2, 200))
    master_list = list(executor.map(normal_MC, np.arange(2, 500)))
    
    
'''
# Convergence test
ST_C = np.zeros(len(range(2, N)))
for idn, n in enumerate(range(2, N)):
    Y = Y_base[:, :n, :, :n, :, :n]
    ST_C[idn] = normal_MC(Y)
    print('n = %4d. PST =  %.4f' %(n, ST_C[idn]))
    
np.save('ST_C_normal.npy', ST_C)

plt.figure(figsize=(6, 3))
plt.plot(np.arange(2, N)**3 * 8, ST_C)
plt.xlabel('Simulation runs')
plt.ylim(0.50, 1.0)
plt.xlim(0, 1e7)
plt.ylabel('$PS_{TK}$' + ' of snowmelt process')
plt.show()
'''







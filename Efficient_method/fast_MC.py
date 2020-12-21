# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:38:47 2020

@author: Jing
"""
import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli

# Global settings
skip_values = 1000
plt.style.use('default')

# Model info
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

# Process model functions
def model_R1(a):
    """
    Compute recharge[m/d] using recharge model R1 by Chaturvedi(1936)
    
    """
    return a * (P - 14)**0.5 * 25.4 * 0.001 / 365

def model_R2(b):
    """
    Compute recharge[m/d] using recharge model R2 by Krishna Rao (1970)
    
    """
    return b * (P - 15.7) * 25.4 * 0.001 / 365

def model_M1(f1):
    """
    Compute river stage h2 [m] using degree-day method
 
    """
    M = f1 * (Ta - Tm)
    Q = Csn * M * SVC * A * 0.001 / 86400
    h2 = 0.3 * Q**0.6 + z0
    
    return h2

def model_M2(f2, r):
    """
    Compute river stage h2 [m] using restricted degree-day radiation balance approach

    """
    M = f2 * (Ta - Tm) + r * Rn
    Q = Csn * M * SVC * A * 0.001 / 86400
    h2 = 0.3 * Q**0.6 + z0
    
    return h2

# Compute the output
def cmpt_Y(w, k1, k2, h2):
    C1 = (h1**2 - h2**2 - w / k1 * x0**2 + w / k2 * x0**2 - w / k2 * L**2) / (k1 / k2 * x0 - k1 / k2 * L - x0)
    return w * x0 - k1 * C1 / 2

# Generate parameters using SALib.sample.saltelli
def generate_param_values(N, skip_values):
    return  saltelli.sample(problem, N, calc_second_order=False, skip_values=skip_values)[::10, :]
    
def fast_MC(nobs):
    # Number of parameter realizations
    N = nobs
    
    # Generate parameter values for matries A and B
    param_values = generate_param_values(N, skip_values)
    param_values_prime = generate_param_values(N, skip_values + N)
    
    # ==============================R1G1M1 ==============================
    # Matrix A
    w = model_R1(param_values[:, 0])
    k1 = param_values[:, 2]
    k2 = param_values[:, 2]
    h2 = model_M1(param_values[:, 5])
    Y_A_R1G1M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix B
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M1(param_values_prime[:, 5])
    Y_B_R1G1M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C1
    w = model_R1(param_values[:, 0])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M1(param_values_prime[:, 5])
    Y_C1_R1G1M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C2
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values[:, 2]
    k2 = param_values[:, 2]
    h2 = model_M1(param_values_prime[:, 5])
    Y_C2_R1G1M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C3
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M1(param_values[:, 5])
    Y_C3_R1G1M1 = cmpt_Y(w, k1, k2, h2)
    
    # ==============================R1G1M2 ==============================
    # Matrix A
    w = model_R1(param_values[:, 0])
    k1 = param_values[:, 2]
    k2 = param_values[:, 2]
    h2 = model_M2(param_values[:, 6], param_values[:, 7])
    Y_A_R1G1M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix B
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_B_R1G1M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C1
    w = model_R1(param_values[:, 0])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_C1_R1G1M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C2
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values[:, 2]
    k2 = param_values[:, 2]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_C2_R1G1M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C3
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M2(param_values[:, 6], param_values[:, 7])
    Y_C3_R1G1M2 = cmpt_Y(w, k1, k2, h2)
    
    # ==============================R1G2M1 ==============================
    # Matrix A
    w = model_R1(param_values[:, 0])
    k1 = param_values[:, 3]
    k2 = param_values[:, 4]
    h2 = model_M1(param_values[:, 5])
    Y_A_R1G2M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix B
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M1(param_values_prime[:, 5])
    Y_B_R1G2M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C1
    w = model_R1(param_values[:, 0])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M1(param_values_prime[:, 5])
    Y_C1_R1G2M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C2
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values[:, 3]
    k2 = param_values[:, 4]
    h2 = model_M1(param_values_prime[:, 5])
    Y_C2_R1G2M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C3
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M1(param_values[:, 5])
    Y_C3_R1G2M1 = cmpt_Y(w, k1, k2, h2)
    
    # ==============================R1G2M2 ==============================
    # Matrix A
    w = model_R1(param_values[:, 0])
    k1 = param_values[:, 3]
    k2 = param_values[:, 4]
    h2 = model_M2(param_values[:, 6], param_values[:, 7])
    Y_A_R1G2M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix B
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_B_R1G2M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C1
    w = model_R1(param_values[:, 0])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_C1_R1G2M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C2
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values[:, 3]
    k2 = param_values[:, 4]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_C2_R1G2M2 = cmpt_Y(w, k1, k2, h2)

    
   # Matrix C3
    w = model_R1(param_values_prime[:, 0])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M2(param_values[:, 6], param_values[:, 7])
    Y_C3_R1G2M2 = cmpt_Y(w, k1, k2, h2)
    
    # ==============================R2G1M1 ==============================
    # Matrix A
    w = model_R2(param_values[:, 1])
    k1 = param_values[:, 2]
    k2 = param_values[:, 2]
    h2 = model_M1(param_values[:, 5])
    Y_A_R2G1M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix B
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M1(param_values_prime[:, 5])
    Y_B_R2G1M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C1
    w = model_R2(param_values[:, 1])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M1(param_values_prime[:, 5])
    Y_C1_R2G1M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C2
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values[:, 2]
    k2 = param_values[:, 2]
    h2 = model_M1(param_values_prime[:, 5])
    Y_C2_R2G1M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C3
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M1(param_values[:, 5])
    Y_C3_R2G1M1 = cmpt_Y(w, k1, k2, h2)
    
    # ==============================R2G1M2 ==============================
    # Matrix A
    w = model_R2(param_values[:, 1])
    k1 = param_values[:, 2]
    k2 = param_values[:, 2]
    h2 = model_M2(param_values[:, 6], param_values[:, 7])
    Y_A_R2G1M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix B
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_B_R2G1M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C1
    w = model_R2(param_values[:, 1])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_C1_R2G1M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C2
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values[:, 2]
    k2 = param_values[:, 2]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_C2_R2G1M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C3
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values_prime[:, 2]
    k2 = param_values_prime[:, 2]
    h2 = model_M2(param_values[:, 6], param_values[:, 7])
    Y_C3_R2G1M2 = cmpt_Y(w, k1, k2, h2)
    
    # ==============================R2G2M1 ==============================
    # Matrix A
    w = model_R2(param_values[:, 1])
    k1 = param_values[:, 3]
    k2 = param_values[:, 4]
    h2 = model_M1(param_values[:, 5])
    Y_A_R2G2M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix B
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M1(param_values_prime[:, 5])
    Y_B_R2G2M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C1
    w = model_R2(param_values[:, 1])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M1(param_values_prime[:, 5])
    Y_C1_R2G2M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C2
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values[:, 3]
    k2 = param_values[:, 4]
    h2 = model_M1(param_values_prime[:, 5])
    Y_C2_R2G2M1 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C3
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M1(param_values[:, 5])
    Y_C3_R2G2M1 = cmpt_Y(w, k1, k2, h2)
    
    # ==============================R2G2M2 ==============================
    # Matrix A
    w = model_R2(param_values[:, 1])
    k1 = param_values[:, 3]
    k2 = param_values[:, 4]
    h2 = model_M2(param_values[:, 6], param_values[:, 7])
    Y_A_R2G2M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix B
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_B_R2G2M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C1
    w = model_R2(param_values[:, 1])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_C1_R2G2M2 = cmpt_Y(w, k1, k2, h2)
    
    # Matrix C2
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values[:, 3]
    k2 = param_values[:, 4]
    h2 = model_M2(param_values_prime[:, 6], param_values_prime[:, 7])
    Y_C2_R2G2M2 = cmpt_Y(w, k1, k2, h2)

    
   # Matrix C3
    w = model_R2(param_values_prime[:, 1])
    k1 = param_values_prime[:, 3]
    k2 = param_values_prime[:, 4]
    h2 = model_M2(param_values[:, 6], param_values[:, 7])
    Y_C3_R2G2M2 = cmpt_Y(w, k1, k2, h2)
    
    E_t_d = np.mean([Y_A_R1G1M1, Y_A_R1G1M2, Y_A_R1G2M1, Y_A_R1G2M2, Y_A_R2G1M1, Y_A_R2G1M2, Y_A_R2G2M1, Y_A_R2G2M2])
    Var_t_d = np.var([Y_A_R1G1M1, Y_A_R1G1M2, Y_A_R1G2M1, Y_A_R1G2M2, Y_A_R2G1M1, Y_A_R2G1M2, Y_A_R2G2M1, Y_A_R2G2M2])
    '''
    # PSTA Total process sensitivy 
    E_A2_G1M1_IT1 = (Y_A_R1G1M1**2 - (Y_C1_R1G1M1 * Y_B_R1G1M1)) * 0.5 + (Y_A_R2G1M1**2 - (Y_C1_R2G1M1 * Y_B_R2G1M1)) * 0.5
    E_A2_G1M2_IT1 = (Y_A_R1G1M2**2 - (Y_C1_R1G1M2 * Y_B_R1G1M2)) * 0.5 + (Y_A_R2G1M2**2 - (Y_C1_R2G1M2 * Y_B_R2G1M2)) * 0.5  
    E_A2_G2M1_IT1 = (Y_A_R1G2M1**2 - (Y_C1_R1G2M1 * Y_B_R1G2M1)) * 0.5 + (Y_A_R2G2M1**2 - (Y_C1_R2G2M1 * Y_B_R2G2M1)) * 0.5
    E_A2_G2M2_IT1 = (Y_A_R1G2M2**2 - (Y_C1_R1G2M2 * Y_B_R1G2M2)) * 0.5 + (Y_A_R2G2M2**2 - (Y_C1_R2G2M2 * Y_B_R2G2M2)) * 0.5
    
    E_A2_G1M1_IT2 = (Y_A_R1G1M1 - (Y_A_R1G1M1 * 0.5 + Y_A_R2G1M1 * 0.5))**2 * 0.5 + (Y_A_R2G1M1 - (Y_A_R1G1M1 * 0.5 + Y_A_R2G1M1 * 0.5))**2 * 0.5
    E_A2_G1M2_IT2 = (Y_A_R1G1M2 - (Y_A_R1G1M2 * 0.5 + Y_A_R2G1M2 * 0.5))**2 * 0.5 + (Y_A_R2G1M2 - (Y_A_R1G1M2 * 0.5 + Y_A_R2G1M2 * 0.5))**2 * 0.5
    E_A2_G2M1_IT2 = (Y_A_R1G2M1 - (Y_A_R1G2M1 * 0.5 + Y_A_R2G2M1 * 0.5))**2 * 0.5 + (Y_A_R2G2M1 - (Y_A_R1G2M1 * 0.5 + Y_A_R2G2M1 * 0.5))**2 * 0.5
    E_A2_G2M2_IT2 = (Y_A_R1G2M2 - (Y_A_R1G2M2 * 0.5 + Y_A_R2G2M2 * 0.5))**2 * 0.5 + (Y_A_R2G2M2 - (Y_A_R1G2M2 * 0.5 + Y_A_R2G2M2 * 0.5))**2 * 0.5
    
    E_A = np.mean((E_A2_G1M1_IT1 + E_A2_G1M1_IT2) * 0.25 + \
                    (E_A2_G1M2_IT1 + E_A2_G1M2_IT2) * 0.25 + \
                        (E_A2_G2M1_IT1 + E_A2_G2M1_IT2) * 0.25 + \
                            (E_A2_G2M2_IT1 + E_A2_G2M2_IT2) * 0.25)
    ST_A =  E_A / (Var_t_d + 1e-20)
    
    # PSTB Total process sensitivy 
    E_B2_R1M1_IT1 = (Y_A_R1G1M1**2 - (Y_C2_R1G1M1 * Y_B_R1G1M1)) * 0.5 + (Y_A_R1G2M1**2 - (Y_C2_R1G2M1 * Y_B_R1G2M1)) * 0.5
    E_B2_R1M2_IT1 = (Y_A_R1G1M2**2 - (Y_C2_R1G1M2 * Y_B_R1G1M2)) * 0.5 + (Y_A_R1G2M2**2 - (Y_C2_R1G2M2 * Y_B_R1G2M2)) * 0.5
    E_B2_R2M1_IT1 = (Y_A_R2G1M1**2 - (Y_C2_R2G1M1 * Y_B_R2G1M1)) * 0.5 + (Y_A_R2G2M1**2 - (Y_C2_R2G2M1 * Y_B_R2G2M1)) * 0.5
    E_B2_R2M2_IT1 = (Y_A_R2G1M2**2 - (Y_C2_R2G1M2 * Y_B_R2G1M2)) * 0.5 + (Y_A_R2G2M2**2 - (Y_C2_R2G2M2 * Y_B_R2G2M2)) * 0.5
    
    E_B2_R1M1_IT2 = (Y_A_R1G1M1 - (Y_A_R1G1M1 * 0.5 + Y_A_R1G2M1 * 0.5))**2 * 0.5 + (Y_A_R1G2M1 - (Y_A_R1G1M1 * 0.5 + Y_A_R1G2M1 * 0.5))**2 * 0.5
    E_B2_R1M2_IT2 = (Y_A_R1G1M2 - (Y_A_R1G1M2 * 0.5 + Y_A_R1G2M2 * 0.5))**2 * 0.5 + (Y_A_R1G2M2 - (Y_A_R1G1M2 * 0.5 + Y_A_R1G2M2 * 0.5))**2 * 0.5
    E_B2_R2M1_IT2 = (Y_A_R2G1M1 - (Y_A_R2G1M1 * 0.5 + Y_A_R2G2M1 * 0.5))**2 * 0.5 + (Y_A_R2G2M1 - (Y_A_R2G1M1 * 0.5 + Y_A_R2G2M1 * 0.5))**2 * 0.5
    E_B2_R2M2_IT2 = (Y_A_R2G1M2 - (Y_A_R2G1M2 * 0.5 + Y_A_R2G2M2 * 0.5))**2 * 0.5 + (Y_A_R2G2M2 - (Y_A_R2G1M2 * 0.5 + Y_A_R2G2M2 * 0.5))**2 * 0.5
    
    E_B = np.mean((E_B2_R1M1_IT1 * 0.5 + E_B2_R1M1_IT2) * 0.25 + \
                    (E_B2_R1M2_IT1 * 0.5 + E_B2_R1M2_IT2) * 0.25 + \
                        (E_B2_R2M1_IT1 * 0.5  + E_B2_R2M1_IT2) * 0.25 + \
                            (E_B2_R2M2_IT1 * 0.5  + E_B2_R2M2_IT2) * 0.25)
    ST_B =  E_B / (Var_t_d + 1e-20)
    '''
    # PSTC Total process sensitivy 
    E_C2_R1G1_IT1 = (Y_A_R1G1M1**2 - (Y_C3_R1G1M1 * Y_B_R1G1M1)) * 0.5  + (Y_A_R1G1M2**2 - (Y_C3_R1G1M2 * Y_B_R1G1M2)) * 0.5
    E_C2_R1G2_IT1 = (Y_A_R1G2M1**2 - (Y_C3_R1G2M1 * Y_B_R1G2M1)) * 0.5  + (Y_A_R1G2M2**2 - (Y_C3_R1G2M2 * Y_B_R1G2M2)) * 0.5
    E_C2_R2G1_IT1 = (Y_A_R2G1M1**2 - (Y_C3_R2G1M1 * Y_B_R2G1M1)) * 0.5  + (Y_A_R2G1M2**2 - (Y_C3_R2G1M2 * Y_B_R2G1M2)) * 0.5
    E_C2_R2G2_IT1 = (Y_A_R2G2M1**2 - (Y_C3_R2G2M1 * Y_B_R2G2M1)) * 0.5  + (Y_A_R2G2M2**2 - (Y_C3_R2G2M2 * Y_B_R2G2M2)) * 0.5
    
    E_C2_R1G1_IT2 = (Y_A_R1G1M1 - (Y_A_R1G1M1 * 0.5 + Y_A_R1G1M2 * 0.5))**2 * 0.5 + (Y_A_R1G1M2 - (Y_A_R1G1M1 * 0.5 + Y_A_R1G1M2 * 0.5))**2 * 0.5
    E_C2_R1G2_IT2 = (Y_A_R1G2M1 - (Y_A_R1G2M1 * 0.5 + Y_A_R1G2M2 * 0.5))**2 * 0.5 + (Y_A_R1G2M2 - (Y_A_R1G2M1 * 0.5 + Y_A_R1G2M2 * 0.5))**2 * 0.5
    E_C2_R2G1_IT2 = (Y_A_R2G1M1 - (Y_A_R2G1M1 * 0.5 + Y_A_R2G1M2 * 0.5))**2 * 0.5 + (Y_A_R2G1M2 - (Y_A_R2G1M1 * 0.5 + Y_A_R2G1M2 * 0.5))**2 * 0.5
    E_C2_R2G2_IT2 = (Y_A_R2G2M1 - (Y_A_R2G2M1 * 0.5 + Y_A_R2G2M2 * 0.5))**2 * 0.5 + (Y_A_R2G2M2 - (Y_A_R2G2M1 * 0.5 + Y_A_R2G2M2 * 0.5))**2 * 0.5
    
    E_C = np.mean((E_C2_R1G1_IT1 * 0.5 + E_C2_R1G1_IT2) * 0.25 + \
                    (E_C2_R1G2_IT1 * 0.5 + E_C2_R1G2_IT2) * 0.25 + \
                        (E_C2_R2G1_IT1 * 0.5  + E_C2_R2G1_IT2) * 0.25 + \
                            (E_C2_R2G2_IT1 * 0.5  + E_C2_R2G2_IT2) * 0.25)
    ST_C =  E_C / (Var_t_d + 1e-20)
    
    return ST_C




'''
ST_C = fast_MC(10000)


N = 250000
ST_C = np.zeros(len(range(2, N, 100)))
for idn, n in enumerate(range(2, N, 100)):
    ST_C[idn] = fast_MC(n)
    print('n = %4d. PST =  %.4f' %(n, ST_C[idn]))

np.save('ST_C_fast_MC.npy', ST_C)

plt.figure()
plt.plot(np.arange(2, N, 100) * 5 * 8, ST_C)
plt.xlabel('Simulation runs')
plt.ylabel('$PS_{TK}$')
#plt.ylim(0.175, 0.225)
plt.show()

'''





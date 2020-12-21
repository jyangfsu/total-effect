# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 10:33:04 2020

@author: Jing
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


plt.style.use('default')

ST_C_normal = np.load('ST_C_normal_500.npy')
ST_C_fast = np.load('ST_C_fast_2500000.npy')
ST_C_fast[-7:-1] = ST_C_fast[-8]

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(np.arange(2, 500)**3 * 8, ST_C_normal * 100, lw=2.0, label='Direct MC method', 
        color='r', alpha=0.8)
ax.plot(np.hstack([np.floor(np.linspace(2, 1e5, 1000, endpoint=False)), 
                   np.floor(np.linspace(1e5, 2.5e6, 10))]) * 5 * 8, 
                   ST_C_fast * 100, lw=2.0,  color='b', alpha=0.8,
                   label='New method')
ax.set_ylim(65, 85)
ax.set_xlim(0, 1e8)
ax.set_xlabel('Number of model excuations',  fontsize=14)
ax.set_ylabel('$PS_{TK}$' + ' of snowmelt process (%)',  fontsize=14)
ax.set_xticks([0.0, 2e7, 4e7, 6e7, 8e7, 1e8]) 
ax.set_xticklabels(['$0$', r'$2\times10^7$', r'$4\times10^7$', r'$6\times10^7$', r'$8\times10^7$', r'$1\times10^8$'], fontsize=13)
ax.set_yticks([65, 70, 75, 80, 85]) 
ax.set_yticklabels(['65', '70', '75', '80', '85'], fontsize=13)
    
ax.legend(frameon=False, fontsize=13)

axin = ax.inset_axes((0.2, 0.2, 0.5, 0.3))
axin.plot(np.arange(2, 500)**3 * 8, ST_C_normal * 100, lw=2.0, label='Brute force method',
          color='r', alpha=0.8)
axin.plot(np.hstack([np.floor(np.linspace(2, 1e5, 1000, endpoint=False)), 
                     np.floor(np.linspace(1e5, 2.5e6, 10))]) * 5 * 8, 
                     ST_C_fast *100, lw=2.0, color='b', alpha=0.8,
                     label='New method')
axin.set_ylim(65, 85)
axin.set_xlim(0, 1e6)
axin.set_yticks([70, 75, 80]) 
axin.set_yticklabels(['70', '75', '80'], fontsize=12)
axin.set_xticks([0.0, 5e5, 1e6]) 
axin.set_xticklabels(['$0$', r'$5\times10^5$', r'$1\times10^6$'], fontsize=12)




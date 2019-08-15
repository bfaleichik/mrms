# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:57:11 2019

@author: Boris Faleichik

A script to perform benchmarking of the MRMS methods
on a model linear diagonal problem
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import problems
import solvers


solvers.verbose = True

# Set do_test to True to perform the numerical experiment,
# otherwise the data for plotting will be taken from mrms_file
# specified below
do_test = True


n = 100
lamin = -100
kshift = 1  # MRMS(k+kshift, k) methods will be used

evals = np.linspace(lamin, 0, n)
# evals = -(10. ** np.linspace(-7, 7, n)
vb = np.ones(n)
y0 = np.ones(n)
# y0 = -vb / evals
t0, tend = 0, 1
ivp = problems.DiagonalLinear_IVP(evals, vb, y0, [t0, tend])
rsol = ivp.rsol(tend)



# The filenames below should be valid in order to store or get the experiment data
mrms_file = 'd:/diag{}+{}-mrms.pcl'.format(lamin, kshift)
picture_file = 'd:/diag{}+{}.pdf'.format(lamin, kshift)

krange = range(1, 8 if lamin == -100 else 7)
steprange = 2 ** np.arange(4, 14)


def test_MRMS(k, p, nsteps):
    def mrms_solver(): return solvers.solve_fixed_mrms(ivp, k, p, nsteps)
    return(solvers.benchmark(mrms_solver, rsol))


def test_Euler(nsteps):
    def euler_solver(): return solvers.solve_fixed_bdf(ivp, 1, nsteps)
    return(solvers.benchmark(euler_solver, rsol))


if do_test:
    mrms_data = []
    for k in krange:
        mrms = np.array([test_MRMS(k+kshift, k, steps) for steps in steprange]).T
        mrms_data.append({'k': k, 'data': mrms})
        
    euler_data = np.array([test_Euler(steps) for steps in steprange]).T


    with open(mrms_file, 'wb') as file:
        pickle.dump(mrms_data, file)
        pickle.dump(euler_data, file)
else:
    with open(mrms_file, 'rb') as file:
        mrms_data = pickle.load(file)
        euler_data = pickle.load(file)
        
           
# PLOTTING

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.gca().tick_params(labelsize=20) 
plt.xlabel('Nsteps', size=20)
plt.ylabel('error', size=20)


lw = 2.  # linewidth
ms = 12.  # markersize

# plt.xlim((1, 2 * 1e2)) # x axis range
for d in mrms_data:
    plt.loglog(steprange, d['data'][1], marker='o', linewidth=lw, markersize=ms,
               label='MRMS({},{})'.format(d['k']+kshift, d['k']))

plt.loglog(steprange, euler_data[1], '--', marker='s', color='gray', linewidth=lw, markersize=ms,
               label='Implicit Euler')

if lamin == -100:
    plt.loglog(steprange, 100/steprange, '-.', color='black', linewidth=lw, label='Order 1')
    plt.loglog(steprange, 100000*(1/steprange)**6, '--', color='black', linewidth=lw, label='Order 6')
    plt.ylim(bottom=1e-13)


plt.ylim(top=10)
# plt.xlim(right=steprange[-1]*2

plt.grid()
legloc = 'lower left' if lamin == -100 else 'upper left'
plt.legend(loc=legloc, prop={'size': 20})
# plt.show()

fig.savefig(picture_file, dpi=100, format='pdf')

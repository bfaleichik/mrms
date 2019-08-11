# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:57:11 2019

@author: faleichik
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import problems
import solvers


solvers.verbose = True
# Set do_test to True to perform the numerical experiment,
# otherwise the data for plotting will be taken from bdf_file and mrms_file
# specified below
do_test = False

# the grid dimension 
N = 20

bdf_file = 'd:/N{}-bdf.pcl'.format(N)
mrms_file = 'd:/N{}-mrms.pcl'.format(N)
picture_file = 'd:/N{}.pdf'.format(N)

(t0, tend) = (0, 10)
krange = range(1, 6)
steprange = 50 * (2 ** np.arange(0, 6))
if N == 1000:
    steprange = 5 * (2 ** np.arange(0, 6))


def test_BDF(p, nsteps):
    def bdf_solver(): return solvers.solve_fixed_bdf(ivp, p, nsteps)
    return(solvers.benchmark(bdf_solver, rsol))


def test_MRMS(k, nsteps):
    def mrms_solver(): return solvers.solve_fixed_mrms(ivp, k, k, nsteps)
    return(solvers.benchmark(mrms_solver, rsol))


if do_test:
    bdf_data = []
    mrms_data = []
    ivp = problems.Heat2D_IVP(N, [t0, tend])
    rsol = ivp.rsol(tend)
    for k in krange:
        bdf = np.array([test_BDF(k, steps) for steps in steprange]).T
        mrms = np.array([test_MRMS(k, steps) for steps in steprange]).T
        bdf_data.append({'N': N, 'k': k, 'data': bdf})
        mrms_data.append({'N': N, 'k': k, 'data': mrms})
    with open(bdf_file, 'wb') as file:
        pickle.dump(bdf_data, file)
    with open(mrms_file, 'wb') as file:
        pickle.dump(mrms_data, file)
else:
    with open(bdf_file, 'rb') as file:
        bdf_data = pickle.load(file)
    with open(mrms_file, 'rb') as file:
        mrms_data = pickle.load(file)


# PLOTTING

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.gca().tick_params(labelsize=20)
plt.xlabel('time', size=20)
plt.ylabel('error', size=20)

lw = 2.  # linewidth
ms = 12.  # markersize

if N == 400:
    plt.xlim((1, 1.5 * 1e2))  # x axis range

if N == 1000:
    plt.xlim((0.7, 1.5 * 1e2))  # x axis range

    
for d in bdf_data:
    plt.loglog(*d['data'], '--', marker='s', linewidth=lw, markersize=0.7*ms, 
               label='BDF-{}'.format(d['k']))
plt.gca().set_prop_cycle(None)    
for d in mrms_data:
    plt.loglog(*d['data'], marker='o', linewidth=lw, markersize=ms,
               label='MRMS({0},{0})'.format(d['k']))

plt.grid()    
plt.legend(loc='lower left', prop={'size': 20})
plt.title('N={}'.format(N), size=20)
#plt.show()

#fig.savefig(picture_file, dpi=100)
fig.savefig(picture_file, dpi=100, format='pdf')







# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:57:11 2019

@author: Boris Faleichik

A script to test the solution of the 2D heat problem
"""

import numpy as np
from scipy.integrate import solve_ivp
import problems
import solvers


solvers.verbose = True

# test problem
N = 200
(t0, tend) = (0, .2)
ivp = problems.Heat2D_IVP(N, [t0, tend])
rsol = ivp.rsol(tend)

p = 5
k = 5
nsteps = 50


def bdf_solver():
    return solvers.solve_fixed_bdf(ivp, p, nsteps)


def mrms_solver():
    return solvers.solve_fixed_mrms(ivp, k, p, nsteps)


def solve_ivp_solver():
    meth = 'RK23'
    tol = 1e-4
    sol = solve_ivp(ivp.getf, ivp.interval, ivp.y0
                                    , jac = ivp.A
                                    , method = meth
                                    , rtol = tol
                                    , atol = tol
                                    , dense_output = False
                                    #, first_step =  (tend-t0)/nsteps
                                    )
    print('solve_ivp: nlu={}, nfev={}'.format(sol.nlu, sol.nfev))
    return sol.y[:,-1]


solvers.benchmark(bdf_solver, rsol)
solvers.benchmark(mrms_solver, rsol)
solvers.benchmark(solve_ivp_solver, rsol)

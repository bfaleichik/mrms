# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:57:11 2019

@author: Boris Faleichik
"""
import numpy as np
from scipy.integrate import solve_ivp
import problems
import solvers


solvers.verbose = True

# test problem
n = 100
# evals = -10 ** np.linspace(-5, 7, n)
evals = np.linspace(-100, 0, n)
vb = np.zeros(n)
y0 = np.ones(n)
# y0 = -vb / evals
t0, tend = 0, 1
ivp = problems.DiagonalLinear_IVP(evals, vb, y0, [t0, tend])
rsol = ivp.rsol(tend)

p = 5
k = 12
nsteps = 4096


def bdf_solver():
    return solvers.solve_fixed_bdf(ivp, p, nsteps)


def mrms_solver():
    return solvers.solve_fixed_mrms(ivp, k, p, nsteps)


def solve_ivp_solver():
    meth = 'BDF'
    tol = 1e-4
    sol = solve_ivp(ivp.getf, ivp.interval, ivp.y0
                                    , jac = ivp.A
                                    , method = meth
                                    , rtol = tol
                                    , atol = tol
                                    , dense_output = False)
    print('solve_ivp: nlu={}, nfev={}'.format(sol.nlu, sol.nfev))
    return sol.y[:, -1]


solvers.benchmark(bdf_solver, rsol)
solvers.benchmark(mrms_solver, rsol)
solvers.benchmark(solve_ivp_solver, rsol)

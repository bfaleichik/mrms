# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:57:11 2019

@author: Boris Faleichik

This file contains MRMS and BDF fixed step solvers
and some auxiliary functions

"""

import numpy as np
# from numpy.linalg import norm as norm
import scipy as sp
from scipy.integrate import solve_ivp

# import math

from problems import *

verbose = True


##########################################################
def lprint(str):
    if verbose:
        print(str)


##########################################################
def norm(v): return np.linalg.norm(v, ord=np.inf)


##########################################################
def getBDF(ts):
    """
        Returns BDF method coefficients which correspond to
        the grid points specified by ts
    """
    k = len(ts)
    a = np.zeros([k, k])
    rhs = np.zeros_like(ts)
    rhs[1] = 1  # ts[-1] - ts[-2]
    a[0] = np.ones_like(ts)

    for i in range(1, k):
        for j in range(0, k-i):
            a[i, j] = a[i-1, j] * (ts[j] - ts[k-i])
        if (i > 1):
            rhs[i] = rhs[i-1] * (ts[k-1] - ts[k-i])

    # can avoid reversion here by constructing the required form of matrix above
    return(sp.linalg.solve_triangular(np.flip(a, 1), rhs)[::-1])


##########################################################
def solve_fixed_mrms(ivp, k, p, nsteps):
    """
        Fixed step Minimal Residual MultiStep integration
        of the problem y'= A y + b(t), matrix A is constant
        
        Input parameters:
            ivp   : an object representing an IVP to solve
            k     : use k-step MRMS method
            p     : use underlying BDF formula of order p
            nsteps: number of steps to perform
    """
    assert k >= p
    assert k <= nsteps
    [t0, tend] = ivp.interval
    tau = (tend - t0) / nsteps
    lprint('MRMS: k={}, p={}, tau={}'.format(k, p, tau))

    cs = getBDF(np.arange(p+1)) / tau
    ck = cs[-1]
    cs = cs[:-1]

    ts = np.array([t0], dtype='float')
    V = np.column_stack((ivp.y0, ivp.getf(t0, ivp.y0)))

    # initialize starting values from the exact solution
    for _ in range(1, k):
        ts = np.append(ts, ts[-1] + tau)
        V = np.column_stack((V, ivp.rsol(ts[-1])))
        V = np.column_stack((V, ivp.getf(ts[-1], V[:, -1])))

    V = np.asarray(V, order='F')
    W = np.asarray(ivp.A @ V - ck * V, order='F')

    lstime = 0

    timer = Timer()
    step = 0
    ptr = 0
    y_id = np.arange(2 * (k-p), 2*k, 2)
    shift_fun = np.vectorize(lambda i: i + 2 if i < 2*k-2 else 0)
    timer.tic()
    for _ in range(k-1, nsteps):
        step += 1
        t1 = ts[-1] + tau
        b = ivp.b(t1)

        g = V[:, y_id].dot(cs) - b
        y_id = shift_fun(y_id)
                
        lstime -= time.clock()
        x, res, rank, s = sp.linalg.lstsq(W, g, lapack_driver='gelsd')
        #print(s[0]/s[-1])
        lstime += time.clock()
        y1 = V.dot(x)


        ts = np.roll(ts, -1)
        ts[-1] = t1

        V[:, ptr] = y1
        Ay1 = ivp.A @ y1
        V[:, ptr + 1] = Ay1 + b
        W[:, ptr] = Ay1 - ck * y1
        W[:, ptr + 1] = ivp.A @ V[:, ptr + 1] - ck * V[:, ptr + 1] 
        ptr += 2
        if ptr >= 2*k: ptr = 0
        pass
    
    timer.toc()
    lprint('MRMS: average step time is {} seconds'.format(timer.elapsed / step))
    lprint('MRMS: average lstsq time={}'.format(lstime / step))
    return y1
    
    
##########################################################
def solve_fixed_bdf(ivp, k, nsteps):
    """
        Fixed step BDF integration
        of the problem y'= A y + b(t), matrix A is constant
        
        Input parameters:
            ivp   : an object representing an IVP to solve
            k     : use k-step BDF method
            nsteps: number of steps to perform
    """
    assert k <= nsteps
    timer = Timer()
    [t0, tend] = ivp.interval
    tau = (tend - t0) / nsteps
    lprint('BDF: k={}, tau={}'.format(k, tau))

    cs = getBDF(np.arange(k+1))

    ts = np.array([t0], dtype='float')
    Y = np.array([ivp.y0], dtype='float')

    for j in range(1, k):
        ts = np.append(ts, ts[-1] + tau)
        Y = np.vstack((Y, ivp.rsol(ts[-1])))

    timer.tic()
    if sp.sparse.issparse(ivp.A):
        M = tau * ivp.A - cs[-1] * sp.sparse.eye(ivp.n)
        lu = sp.sparse.linalg.splu(M)
        lusolve = lu.solve
    else:
        M = tau * ivp.A - cs[-1] * np.eye(ivp.n)
        lu = sp.linalg.lu_factor(M)
        lusolve = lambda b: sp.linalg.lu_solve(lu, b)
    timer.toc()
    lprint("BDF: LU decomposition took {} seconds".format(timer.elapsed))
    
    timer.tic()
    stime = 0
    step = 0  
    ptr = 0
    cs = cs[:-1]
    for _ in range(k-1, nsteps):
        step += 1
        t1 = ts[-1] + tau
        g = (Y.T).dot(cs) - tau * ivp.b(t1) 
        stime -= time.clock()
        y1 = lusolve(g)
        stime += time.clock()
    
        #Y = np.roll(Y, -1, axis=0)#[2:]
        Y[ptr] = y1
        ptr += 1
        if ptr >= k: ptr = 0
        ts = np.roll(ts, -1)#ts[1:]
        cs = np.roll(cs, 1)
        ts[-1] = t1
    timer.toc()
    lprint('BDF: average step time is {} seconds'.format(timer.elapsed / step))
    lprint('BDF: average lusolve time is {} seconds'.format(stime / step))
    return y1
  
##########################################################
def benchmark(solver, rsol):
    """
        A wrapper to measure execution time and error
    """
    lprint('>>> Launching function {}'.format(solver.__name__))
    timer = Timer()
    timer.tic()
    y1 = solver()
    timer.toc()
    err = norm(y1 - rsol)
    lprint('=== Solution took {} seconds, error is {}'.format(timer.elapsed, err))    
    return([timer.elapsed, err])
    

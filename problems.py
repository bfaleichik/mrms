# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:29:16 2019

@author: Boris Faleichik

This file contains classes describing initial value problems for linear ODEs
and some auxiliary functions and classes
"""
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.sparse import linalg as splinalg
import math
import time

######################################################################
# Matrix generators


def diag_sparse_matrix(diag):
    return sp.sparse.diags(diag, format='csc')


def dense_random_matrix(diag):
    n = len(diag)
    X = np.random.rand(n, n)
    return sp.linalg.solve(X, (diag * X))


def do_random_rotation(m):
    n = m.shape[0]
    i, j = tuple(np.random.choice(np.arange(n), size=2))
    a = 2 * math.pi * np.random.rand()
    idi = idj = np.array(range(n))
    vals = np.ones(n, dtype = float)
    vals[i] = vals[j] = math.cos(a)
    vals = np.append(vals, [math.sin(a), -math.sin(a)])
    idi = np.append(idi, [i, j])
    idj = np.append(idj, [j, i])

    r = sp.sparse.csc_matrix((vals, (idi, idj)), shape=(n, n))
    return (r.T @ m) @ r


def sparse_random_matrix_normal(diag, nrot=None):
    m = sp.sparse.diags(diag, format='csc')
    if nrot == None:
        nrot = round(len(diag) / 5)
    for _ in range(nrot):
        m = do_random_rotation(m)
    return sp.sparse.csc_matrix(m)


def sparse_random_matrix(diag, nnz=None, dfactor=None):
    n = len(diag)
    if nnz is None: nnz = 2 * n
    if dfactor is None: dfactor = 10.
    idi, idj = np.random.choice(np.arange(n), (2, nnz))
    vals = np.random.rand(nnz)
    r = sp.sparse.csc_matrix((vals, (idi, idj)), shape=(n, n)) + dfactor * sp.sparse.eye(n)
    return sp.sparse.csc_matrix(
            splinalg.spsolve(r, sp.sparse.diags(diag, format='csc') @ r))


def laplacian_2D_matrix(N):
    n = N**2
    diag1 = np.ones(n - N)
    diag4 = 4 * np.ones(n)
    diag10 = np.ones(n-1)
    diag10[N-1:n:N] = 0
    return sp.sparse.diags((-diag1, -diag10, diag4, -diag10, -diag1),
                           (-N, -1, 0, 1, N), format='csc')


######################################################################
class Timer:

    def tic(self):
        self.start = time.clock()

    def toc(self):
        self.stop = time.clock()
        self.elapsed = self.stop - self.start


######################################################################
# IVP classes

class DiagonalLinear_IVP:

    def __init__(self, diagonal, vb, y0, interval):
        """
        Initialize diagonal linear IVP of the form y' = A y + b with constant diagonal matrix A and constant vector b

        Input parameters:

            diagonal: the diagonal of matrix A
            vb      : vector b
            y0      : initial value y(t0)
            interval: integration interval [t0, tend]
        """
        assert len(diagonal) == len(vb) == len(y0)
        self.n = len(y0)
        self.A = diag_sparse_matrix(diagonal)
        self.diagonal = diagonal
        self.vb = vb
        self.y0 = y0
        self.interval = interval
        self.t0, self.tend = interval
        
        def rsolfun(t, lam, b, y0):
            if lam != 0: 
                res = math.exp(lam * (t - self.t0)) * (y0 + b/lam) - b/lam 
            else:
                res = y0 + b * (t - self.t0)
            return res
            
        self.rsolfun = np.vectorize(rsolfun)

    def b(self, t):  # vector b
        return self.vb
        
    def getf(self, t, y):  # ODE right-hand side
        return self.A @ y + self.vb

    def rsol(self, t):  # exact solution
        evals = self.diagonal
        b = self.vb
#        return (np.exp(evals * t) * (self.y0 + b/evals) - b/evals)
        return self.rsolfun(t, evals, b, self.y0)
        
######################################################################
#class ModelLinearIVP:
#
#    def __init__(self, A, b, y0, interval, tol=1e-12, meth='BDF'):
#        assert len(b(interval[1])) == len(y0) == A.shape[1] == A.shape[0]
#        self.n = len(y0)
#        self.A = A
#        self.b = b
#        self.y0 = np.array(y0, order='F')
#        self.interval = interval
#        timer = Timer()
#        timer.tic()
#        if meth is not None:
#            self.sol = solve_ivp(self.getf, interval, y0
#                                    , jac = A
#                                    , method = meth
#                                    , rtol = tol
#                                    , atol = tol
#                                    , dense_output = True
#                                )
#            timer.toc()
#            print('solve_ivp took {} seconds'.format(timer.elapsed))
#        else:
#            self.sol = None
#
#    def getf(self, t, y):
#        return self.A @ y + self.b(t)
#
#    def rsol(self, t):
#        return self.sol.sol(t) if self.sol is not None else self.y0


######################################################################
class Heat2D_IVP:
    """
    A heat 2D problem with preset exact solution
    """
    def __init__(self, N, interval):
        h = 1 / (N+1)
        self.A = -laplacian_2D_matrix(N) / (h * h)
        grid = np.linspace(h, 1 - h, N)
        xx, yy = np.meshgrid(grid, grid, sparse=True)
        a1, a2, a3 = 1, 2, 3
        u2xy = np.exp(xx + yy) * (np.sin(a2 * math.pi * xx) *
                      np.sin(a3 * math.pi * yy))
#        u2xy = np.exp(xx + yy) * xx * (1 - xx) * yy * (1 - yy)

        self.u2xy = u2xy.flatten()
        self.h = h
        self.N = N
        self.n = N**2
        self.interval = interval
        self.y0 = self.rsol(interval[0])
        self.Au2 = self.A @ self.u2xy
        pass

    def rsol(self, t):   # exact solution
        return (1 + math.cos(t)) * self.u2xy

    def d_rsol_dt(self, t):  # time derivative of the exact solution
        return (-math.sin(t)) * self.u2xy

    def b(self, t):  # vector b(t) matching the exact solution
        return self.d_rsol_dt(t) - (1 + math.cos(t)) * self.Au2

    def getf(self, t, y):  # MOL ODE right-hand side
        return self.A @ y + self.b(t)

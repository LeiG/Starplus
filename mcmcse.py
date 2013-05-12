#!/usr/bin/python

"""
Monte Carlo Standard Errors for MCMC

mcse: Compute Monte Carlo standard errors for expectations(univariate and multivariate)
"""

from inspect import isfunction
from numpy import *

def mcse(x, size = "sqroot", g = None, method = "bm"):
    y = asarray(x)  # transform x to array
    
    # pre-inspections
    if not isfunction(g):
        def g(y):   # if g is not a function, return x
            return y
    g = vectorize(g)
    if y.ndim == 1:
        length = y.shape[0]
        dim = 1
    elif y.ndim == 2:
        length = y.shape[1]
        dim = y.shape[0]
    else:
        raise ValueError("x must be 2D matrix!")
    if length < 100: # too few samples
        raise ValueError("Too few samples!")
        
    # methods for size
    if size == "sqroot":
        b = floor(sqrt(length))
        a = floor(length/b)
    elif size == "cuberoot":
        b = floor(length**(1./3.))
        a = floor(length/b)
    else:
        raise ValueError("Wrong size!")
        
    # methods for batch means
    if method == "bm":
        batch = zeros((dim, a))
        mu_hat = zeros((dim, 1))
        var_hat = zeros((dim, 1))
        se_hat = zeros((dim, 1))
        for d in xrange(dim):
            for k in xrange(int(a)):
                batch[d, k] = mean(g(y[d, (k*b):((k+1)*b)]))
            mu_hat[d, 0] = mean(batch[d, ])
            var_hat[d, 0] = b*sum((batch[d, ] - mu_hat[d, ])**2)/(a - 1)
            se_hat[d, 0] = sqrt(var_hat[d, 0])
        return mu_hat, se_hat
    if method == "obm":
        a = length - b + 1
        batch = zeros((dim, a))
        mu_hat = zeros((dim, 1))
        var_hat = zeros((dim, 1))
        se_hat = zeros((dim, 1))
        for d in xrange(dim):
            for k in xrange(int(a)):
                batch[d, k] = mean(g(y[d, k:(k+b)]))
            mu_hat[d, 0] = mean(g(y))
            var_hat[d, 0] = length*b*sum((batch[d, ] - mu_hat[d, ])**2)/(a - 1)/a
            se_hat[d, 0] = sqrt(var_hat[d, 0])
        return mu_hat, se_hat

    
    
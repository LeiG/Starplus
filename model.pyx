#!/usr/bin/python

'''
Bayesian variable selection model attributes

design(tp, press): design matrix
cov(tp, rho): covariance matrix
S: S function
'''

#cython: boundscheck=False
#cython: wraparound=False

from __future__ import division
import numpy as np
cimport numpy as np

from numpy.linalg import inv
from scipy.special import gamma as Gamma


#### design matrix ####
# initial stimulus
cpdef np.ndarray[double, ndim = 1] sti_0(np.ndarray[double, ndim = 1] t):
    
    cdef int l = t.shape[0]
    cdef int tt
    cdef np.ndarray output = np.zeros(l, dtype = np.float)
    
    for tt from 0 <= tt < l:
        if t[tt] == 0.0:
            output[tt] = 1.0
        else:
            output[tt] = 0.0
    return output        

# first stimulus
cpdef np.ndarray[double, ndim = 1] sti_1(np.ndarray[double, ndim = 1] t):

    cdef int l = t.shape[0]
    cdef int tt
    cdef np.ndarray output = np.zeros(l, dtype = np.float)
    
    for tt from 0 <= tt < l:
        if t[tt] <= 4.0 and t[tt] >= 0.0:
            output[tt] = 1.0
        else:
            output[tt] = 0.0
    return output
        
# second stimulus
cpdef np.ndarray[double, ndim = 1] sti_2(np.ndarray[double, ndim = 1] t, double press):

    cdef double t_press = min(12.0, press)
    cdef int l = t.shape[0]
    cdef int tt
    cdef np.ndarray output = np.zeros(l, dtype = np.float)
    
    for tt from 0 <= tt < l:
        if t[tt] > 8.0 and t[tt] <= t_press:
            output[tt] = 1.0
        else:
            output[tt] = 0.0
    return output

# HRF function scaled such that sum(hrf(t))=1 for t = np.arange(0, 27, 0.5)
cpdef np.ndarray[double, ndim = 1] hrf(np.ndarray[double, ndim = 1] t):

    cdef int l = t.shape[0]
    cdef int tt
    cdef np.ndarray output = np.zeros(l, dtype = np.float)
    
    for tt from 0 <= tt < l:
        if t[tt] < 0.0:
            output[tt] = 0.0
        else:
            #return 1.0/111.8*(t**8.60)*np.exp(-t/0.547)
            output[tt] = 5.6999155101700625*((t**5)*np.exp(-t)/Gamma(6.0)-1/6.0*(t**15)*np.exp(-t)/Gamma(16.0))/9.5187445708326752                
    return output

cpdef np.ndarray[double, ndim = 1] conv(np.ndarray[double, ndim = 1] hrf, np.ndarray[double, ndim = 1] sti):  # convolved impulse
    
    cdef np.ndarray[double, ndim = 1] output = range(np.size(sti))
    cdef int value, i
    
    for value in output:
        output[value] = 0
        for i from 0 <= i < value+1:
            output[value] += hrf[i]*sti[value - i]
    return output

# time dependent design matrix
cpdef np.ndarray[double, ndim = 2] design(double tp, double press):
    
    cdef np.ndarray[double, ndim = 1] t = np.arange(0, tp/2.0, 0.5) # time sequence
    cdef np.ndarray[double, ndim = 2] output
  
    output = np.ones((tp, 4))
    output[:, 1] = conv(hrf(t), sti_0(t))
    output[:, 2] = conv(hrf(t), sti_1(t))
    output[:, 3] = conv(hrf(t), sti_2(t, press))
    return output


#### covariance matrix capital gamma ####
cpdef np.ndarray[double, ndim = 2] cov(double tp, double rho):    
    
    cdef int i, j
    cdef int tp_int = np.int(tp)
    cdef np.ndarray output = np.zeros([tp, tp], dtype = np.float)
    
    for i from 0 <= i < tp_int:
        for j from 0 <= j < tp_int:
            output[i, j] = rho**np.fabs(i - j)
    
#     output = [[rho**np.fabs(i - j) for i in range(tp)] for j in range(tp)]   # AR(1) structure
    return output
    
cpdef tuple cov_matrix(np.ndarray[double, ndim = 1] rho, int N, double tp):
    
    cdef int v
    cdef np.ndarray[double, ndim = 1] t = np.arange(0, tp/2.0, 0.5) # time sequence
    cdef dict output = {}
    cdef dict output_inv = {}
    
    for v from 0 <= v < N:
        output.update({v: cov(tp, rho[v])})
        output_inv.update({v: inv(cov(tp, rho[v]))})
    return output, output_inv
    
    
#### Ising prior ####
# weight of interaction between voxels
cpdef double w(np.ndarray[double, ndim = 2] coord, int v, int k):  

    cdef double dis
      
    dis = np.sqrt((coord[v][0]-coord[k][0])**2+(coord[v][1]-coord[k][1])**2+(coord[v][2]-coord[k][2])**2)  # distance between voxels
    if dis == 0.:
        return 0.
    else:
        return 1/dis    # weight set to the reciprocal of distance

# neighbors of voxel v    
cpdef np.ndarray[long, ndim = 1] neig(int v, np.ndarray[double, ndim = 2] coord, int N):  

    cdef int i
    cdef np.ndarray[long, ndim = 1] output
 
    output = np.asarray([i for i from 0 <= i < N if w(coord, v, i) == 1.])
    return output
    
# neighborhood structure
cpdef dict neighbor(int N, np.ndarray[double, ndim = 2] coord):
   
    cdef int v
    cdef dict output = {}  # dictionary for neighborhood structure
   
    for v from 0 <= v < N:
        output.update({v: neig(v, coord, N)})
    return output
    
# log of unnormalized Ising prior
cpdef double log_Ising(int j, np.ndarray[double, ndim = 1] theta, np.ndarray[double, ndim = 2] gamma, dict neigh, np.ndarray[double, ndim = 2] coord, int N):
    
    cdef double s = 0
    cdef int v, k
    cdef double output
    
    for v from 0 <= v < N:
        for k in neigh[v]:
            if gamma[k, j] == gamma[v, j]:
                s += w(coord, v, k)
    
    output = theta[j]*s
    return output
    
    
#### MLE for rho ####
# log likelihood function for AR(1) model
def loglike_ar(np.ndarray[double, ndim = 1] x, np.ndarray[double, ndim = 1] y):

    cdef double rho = x[0]
    cdef double sig = x[1]
    cdef int T_int = len(y) # number of time point
    cdef double T = np.float(T_int)
    cdef double part = 0
    cdef int i

    if rho <= -1 or rho >= 1 or sig <= 0:
        return -np.inf
    else:
        for i from 0 <= i < (T_int-1):
            part += (y[i+1]-rho*y[i])**2.0
        return -T*np.log(2.0*np.pi)/2.0-(T-1.0)*np.log(sig*(1.0-rho**2.0))/2.0-part/(2.0*sig*(1.0-rho**2.0))-np.log(sig)/2.0-y[0]**2.0/(2.0*sig)   # log-likelihood

# derivative of log likelihood for AR(1) model
def loglike_ar_der(np.ndarray[double, ndim = 1] x, np.ndarray[double, ndim = 1] y):   

    cdef double rho = x[0]
    cdef double sig = x[1]
    cdef int T_int = len(y) # number of time point
    cdef double T = np.float(T_int)
    cdef double part_1 = 0
    cdef double part_2 = 0
    cdef double der_rho, der_sig
    cdef int i
    
    for i from 0 <= i < (T_int-1):
        part_1 += (y[i+1] - rho*y[i])**2.0
        part_2 += (y[i+1] - rho*y[i])*y[i]
    
    der_rho = (T-1.0)*rho/(1.0-rho**2.0) - rho*part_1/(sig*(1.0-rho**2.0)**2.0) + part_2/(sig*(1.0-rho**2.0))
    der_sig = -(T-1.0)/(2.0*sig) + part_1/(2.0*(1.0-rho**2.0)*(sig**2.0)) - 1.0/(2.0*sig) + y[0]**2.0/(2.0*(sig**2.0))
    return [der_rho, der_sig]

def loglike_ar_hess(np.ndarray[double, ndim = 1] x, np.ndarray[double, ndim = 1] y):  # hessian matrix of log likelihood for AR(1) model
    
    cdef double rho = x[0]
    cdef double sig = x[1]
    cdef int T_int = len(y) # number of time point
    cdef double T = np.float(T_int)
    cdef double part_1 = 0.0
    cdef double part_2 = 0.0
    cdef double part_3 = 0.0
    cdef int i
    cdef np.ndarray[double, ndim = 2] hess = np.zeros((2, 2))

    for i from 0 <= i < (T_int-1):
        part_1 += (y[i+1]-rho*y[i])**2.0
        part_2 += (y[i+1]-rho*y[i])*y[i]
        part_3 += y[i]**2.0
    
    hess[0, 0] = (T-1.0)*(1.0+rho**2.0)/((1.0-rho**2.0)**2.0) - (1.0+3.0*rho**2.0)*part_1/(sig*(1.0-rho**2.0)**3.0) + 4.0*rho*part_2/(sig*(1.0-rho**2.0)**2.0) - part_3/(sig*(1.0-rho**2.0))
    hess[0, 1] = rho*part_1/(((1.0-rho**2.0)**2.0)*(sig**2.0)) - part_2/((sig**2.0)*(1.0-rho**2.0))
    hess[1, 0] = hess[0, 1]
    hess[1, 1] = (T-1.0)/(2.0*sig**2.0) - part_1/((1.0-rho**2.0)*(sig**3.0)) + 1.0/(2.0*sig**2.0) - y[0]**2.0/(sig**3.0)
    return hess
    
cpdef np.ndarray[double, ndim = 1] Newton(f, f_der, f_hess, np.ndarray[double, ndim = 1] x_0, np.ndarray[double, ndim = 1] data, double eps = 10**(-6)):    # Newton's method    
    
    cdef double f_0 = f(x_0, data)  # benchmark
    cdef np.ndarray[double, ndim = 1] y_0 = x_0
    
#     import pdb; pdb.set_trace() # debug flag
    y_0 -= np.dot(inv(f_hess(y_0, data)), f_der(y_0, data))    # update
    while abs(f(y_0, data) - f_0) > eps:
        f_0 = f(y_0, data)  # benchmark
        y_0 -= np.dot(inv(f_hess(y_0, data)), f_der(y_0, data))
    return y_0
    
# MLE for rho and sigma 
cpdef np.ndarray[double, ndim = 2] rhosig_mle(np.ndarray[double, ndim = 2] data, int N):
    
    cdef int v
    cdef np.ndarray[double, ndim = 2] output
    
    output = np.array([Newton(loglike_ar, loglike_ar_der, loglike_ar_hess, np.array([0.0, 1.0]), data[:, v]) for v from 0 <= v < N])
    return output
    
    
#### S function ####
# beta hat
cpdef np.ndarray[double, ndim = 2] beta_hat(int v, dict cov_inv, np.ndarray[int, ndim = 2] gamma, np.ndarray[double, ndim = 2] data, double tp, np.ndarray[double, ndim = 2] design):
    
    cdef np.ndarray[double, ndim = 2] y, d_nonzero, output_p1, output_p2, output
    
    y = data[:, v].reshape(tp, 1)
    d_nonzero = design[:,np.nonzero(gamma[v,:])[0]] # design matrix nonzero part
    output_p1 = (d_nonzero.T.dot(cov_inv[v])).dot(d_nonzero)
    output_p2 = (d_nonzero.T.dot(cov_inv[v])).dot(y)
    output = inv(output_p1).dot(output_p2)
    return output

cpdef np.ndarray[double, ndim = 2] S(int v, dict cov_inv, np.ndarray[int, ndim = 2] gamma, np.ndarray[double, ndim = 2] data, double tp, np.ndarray[double, ndim = 2] design):
    
    cdef np.ndarray[double, ndim = 2] y, d_nonzero, beta, output
    
    y = data[:, v].reshape(tp, 1)
    d_nonzero = design[:,np.nonzero(gamma[v,:])[0]] # design matrix nonzero part
    beta = beta_hat(v, cov_inv, gamma, data, tp, design)    # beta hat
    output = ((y-d_nonzero.dot(beta)).T.dot(cov_inv[v])).dot(y-d_nonzero.dot(beta))
    return output
    
    
    

    



#!/usr/bin/python

'''
Bayesian variable selection model attributes

design(tp, press): design matrix
cov(tp, rho): covariance matrix
S: S function
'''

import numpy as np
from numpy.linalg import inv
from scipy.special import gamma as Gamma


#### design matrix ####
# initial stimulus
def sti_0(t):   
    if t == 0:
        return 1.0
    else:
        return 0.0        
sti_0 = np.vectorize(sti_0) # vectorize

# first stimulus
def sti_1(t):
    if t <= 4 and t >= 0:
        return 1.0
    if t > 4 and t <= 27:
        return 0.0        
sti_1 = np.vectorize(sti_1)   # vectorize
        
# second stimulus
def sti_2(t, press):
    t_press = min(12, press)
    if t > 8 and t <= t_press:
        return 1.0
    if t <= 8 and t >= 0:
        return 0.0
    if t > t_press and t <= 27:
        return 0.0        
sti_2 = np.vectorize(sti_2)   # vectorize

# HRF function scaled such that sum(hrf(t))=1 for t = np.arange(0, 27, 0.5)
def hrf(t):
    if t < 0:
        return 0.0
    else:
        #return 1.0/111.8*(t**8.60)*np.exp(-t/0.547)
        return 5.6999155101700625*((t**5)*np.exp(-t)/Gamma(6.0)-1/6.0*(t**15)*np.exp(-t)/Gamma(16.0))/9.5187445708326752                
hrf = np.vectorize(hrf) # vectorize

def conv(hrf, sti):  # convolved impulse
    output = range(np.size(sti))
    for value in output:
        output[value] = 0
        for i in xrange(value+1):
            output[value] += hrf[i]*sti[value - i]
    return output

# time dependent design matrix
def design(tp, press):  
    t = np.arange(0, tp/2.0, 0.5) # time sequence
    output = np.ones((2*tp, 4))
    output[:, 1] = conv(hrf(t), sti_0(t))
    output[:, 2] = conv(hrf(t), sti_1(t))
    output[:, 3] = conv(hrf(t), sti_2(t, press))
    return output


#### covariance matrix capital gamma ####
def cov(tp, rho):    
    output = [[rho**np.fabs(i - j) for i in range(2*tp)] for j in range(2*tp)]   # AR(1) structure
    output = np.array(output)
    return output
    
def cov_matrix(rho, N, tp):
    t = np.arange(0, tp/2.0, 0.5) # time sequence
    output = {}
    for v in xrange(N):
        output.update({v: cov(tp, rho[v])})
    return output
    
    
#### Ising prior ####
# weight of interaction between voxels
def w(coord, v, k):    
    dis = np.sqrt((coord[v][0]-coord[k][0])**2+(coord[v][1]-coord[k][1])**2+(coord[v][2]-coord[k][2])**2)  # distance between voxels
    if dis == 0.:
        return 0.
    else:
        return 1/dis    # weight set to the reciprocal of distance

# neighbors of voxel v    
def neig(v, coord, N):   
    output = [i for i in range(N) if w(coord, v, i) == 1.]
    return output
    
# neighborhood structure
def neighbor(N, coord):
    output = {}  # dictionary for neighborhood structure
    for v in range(N):
        output.update({v: neig(v, coord, N)})
    return output
    
# log of unnormalized Ising prior
def log_Ising(j, theta, gamma, neigh, coord, N):
    s = np.sum([w(coord, v, k) for v in xrange(N) for k in neigh[v] if gamma[k, j] == gamma[v, j]])
    output = theta[j]*s
    return output
    
    
#### MLE for rho ####
# log likelihood function for AR(1) model
def loglike_ar(x, y):
    rho = x[0]
    sig = x[1]
    T = len(y)  # number of time point
    if rho <= -1 or rho >= 1 or sig <= 0:
        return -inf
    else:
        T = len(y)  # number of time point
        part = np.sum([(y[i+1]-rho*y[i])**2 for i in xrange(T-1)])
        return -T*np.log(2*np.pi)/2-(T-1)*np.log(sig*(1-rho**2))/2-part/(2*sig*(1-rho**2))-np.log(sig)/2-y[0]**2/(2*sig)   # log-likelihood

# derivative of log likelihood for AR(1) model
def loglike_ar_der(x, y):   
    rho = x[0]
    sig = x[1]
    T = len(y)  # number of time point
    der_rho = []
    der_sig = []
    part_1 = np.sum([(y[i+1] - rho*y[i])**2 for i in xrange(T-1)])
    part_2 = np.sum([(y[i+1] - rho*y[i])*y[i] for i in xrange(T-1)])
    der_rho = (T-1)*rho/(1-rho**2) - rho*part_1/(sig*(1-rho**2)**2) + part_2/(sig*(1-rho**2))
    der_sig = -(T-1)/(2*sig) + part_1/(2*(1-rho**2)*(sig**2)) - 1/(2*sig) + y[0]**2/(2*(sig**2))
    return [der_rho, der_sig]

def loglike_ar_hess(x, y):  # hessian matrix of log likelihood for AR(1) model
    rho = x[0]
    sig = x[1]
    T = len(y)  # number of time point
    hess = np.zeros((2, 2))
    part_1 = np.sum([(y[i+1]-rho*y[i])**2 for i in xrange(T-1)])
    part_2 = np.sum([(y[i+1]-rho*y[i])*y[i] for i in xrange(T-1)])
    part_3 = np.sum([y[i]**2 for i in xrange(T-1)])
    hess[0, 0] = (T-1)*(1+rho**2)/((1-rho**2)**2) - (1+3*rho**2)*part_1/(sig*(1-rho**2)**3) + 4*rho*part_2/(sig*(1-rho**2)**2) - part_3/(sig*(1-rho**2))
    hess[0, 1] = rho*part_1/(((1-rho**2)**2)*(sig**2)) - part_2/((sig**2)*(1-rho**2))
    hess[1, 0] = hess[0, 1]
    hess[1, 1] = (T-1)/(2*sig**2) - part_1/((1-rho**2)*(sig**3)) + 1/(2*sig**2) - y[0]**2/(sig**3)
    return hess
    
def Newton(f, f_der, f_hess, x_0, data, eps = 10**(-6)):    # Newton's method
    f_0 = f(x_0, data)  # benchmark
    x_0 -= np.dot(inv(f_hess(x_0, data)), f_der(x_0, data))    # update
    while abs(f(x_0, data) - f_0) > eps:
        f_0 = f(x_0, data)  # benchmark
        x_0 -= np.dot(inv(f_hess(x_0, data)), f_der(x_0, data))
    return x_0
    
# MLE for rho and sigma 
def rhosig_mle(data, N):
    output = np.array([Newton(loglike_ar, loglike_ar_der, loglike_ar_hess, [0,1], data[:, v]) for v in xrange(N)])
    return output
    
    
#### S function ####
# beta hat
def beta_hat(v, cov_inv, gamma, data, tp, design):
    y = data[:, v].reshape(tp, 1)
    d_nonzero = design[:,np.nonzero(gamma[v,:])[0]] # design matrix nonzero part
    output_p1 = (d_nonzero.T.dot(cov_inv)).dot(d_nonzero)
    output_p2 = (d_nonzero.T.dot(cov_inv)).dot(y)
    output = inv(output_p1).dot(output_p2)
    return output

def S(v, cov_inv, gamma, data, tp, design):
    y = data[:, v].reshape(tp, 1)
    d_nonzero = design[:,np.nonzero(gamma[v,:])[0]] # design matrix nonzero part
    beta = beta_hat(v, cov_inv, gamma, data, tp, design)    # beta hat
    output = ((y-d_nonzero.dot(beta)).T.dot(cov_inv)).dot(y-d_nonzero.dot(beta))
    return output
    
    
    

    



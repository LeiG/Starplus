#!/usr/bin/python

'''
MCMC updates for parameters with stopping rule

ratio_theta(j, theta, theta_star): hastings ratio for theta
ratio_gamma(v, j, gamma, gamma_star): hastings ratio for gamma
update_theta(j, theta_cur): stepwise update for theta
update_gamma(v, j, gamma_cur): stepwise update for gamma
mcmc_update(theta, gamma): mcmc updates with stopping rule 
'''

#cython: boundscheck=False
#cython: wraparound=False

# import module
from __future__ import division
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import model

cimport numpy as np

import pickle
from scipy.stats import norm

import model


#### mcmcse.mcse function ####
cpdef tuple mcse(np.ndarray[double, ndim = 2] y):
    
    cdef int length = y.shape[1]
    cdef int dim = y.shape[0]
    cdef int b = np.int(np.sqrt(length))
    cdef int a = length//b
    cdef int d, k
    cdef np.ndarray batch = np.zeros([dim, a], dtype = np.float)
    cdef np.ndarray mu_hat = np.zeros([dim, 1], dtype = np.float)
    cdef np.ndarray se_hat = np.zeros([dim, 1], dtype = np.float)    
    
    for d from 0 <= d < dim:
        for k from 0 <= k < a:
            batch[d, k] = np.mean(y[d, (k*b):((k+1)*b)]) 
        mu_hat[d] = np.mean(batch[d])
        se_hat[d] = np.sqrt(b*np.sum((batch[d] - mu_hat[d])**2)/(a - 1))
        
    return mu_hat, se_hat


#### proposal distribution for gamma ####
# proposal probability
cpdef double gamma_prop(int v, int j, np.ndarray[double, ndim = 2] gamma, np.ndarray[double, ndim = 1] theta, np.ndarray[double, ndim = 2] coord, dict neigh):
    
    cdef int k
    cdef double part, weight, output
    
    for k in neigh[v]:
        weight = model.w(coord, v, k)
        part += weight*(1-2*gamma[k, j])
    
#     part = np.sum([model.w(coord, v, k)*(1-2*gamma[k, j]) for k in neigh])

    output = 1/(1 + np.exp(theta[j]*part))
    return output


#### hastings ratio for theta & gamma ####
# ratio theta
cpdef double ratio_theta(int j, np.ndarray[double, ndim = 1] theta, np.ndarray[double, ndim = 1] theta_star, np.ndarray[double, ndim = 2] gamma, dict neigh, np.ndarray[double, ndim = 2] coord, int N):
    
    cdef double log_output, output
    
    log_output = model.log_Ising(j, theta_star-theta, gamma, neigh, coord, N)+np.log(norm.pdf(theta[j], theta[j], 0.6)/norm.pdf(theta_star[j], theta[j], 0.6))
    if log_output > 0:
        output = 1
    else:
        output = np.exp(log_output)
    return output
    
# ratio gamma
cpdef double ratio_gamma(int v, int j, np.ndarray[double, ndim = 2] gamma, np.ndarray[double, ndim = 2] gamma_star, dict cov_m_inv, np.ndarray[double, ndim = 2] data, double tp, np.ndarray[double, ndim = 2] design_m): # Hastings ratio for updating gamma
    
    cdef double S_1, S_2
    
    S_1 = model.S(v, cov_m_inv, gamma_star, data, tp, design_m)
    S_2 = model.S(v, cov_m_inv, gamma, data, tp, design_m)
    return (1+tp)**(-gamma_star[v, j]/2+gamma[v, j]/2)*(S_1/S_2)**(tp/2)
    

#### stepwise update for theta & gamma ####
# update theta
# accept = [] # record acceptance ratio    
cpdef np.ndarray[double, ndim = 1] update_theta(int j, np.ndarray[double, ndim = 1] theta_cur, np.ndarray[double, ndim = 2] gamma, dict neigh, np.ndarray[double, ndim = 2] coord, int N, char dirname):
    
    cdef int theta_max = 2   # theta_max
    cdef double r, u
    cdef np.ndarray[double, ndim = 1] cur, temp
    
    cur = np.copy(theta_cur)
    temp = np.copy(theta_cur)
    temp[j] = np.random.normal(cur[j], 0.6)  # generate proposal r.v.
    if cur[j] > theta_max or cur[j] < 0:
        cur[j] = temp[j]
    elif temp[j] > theta_max or temp[j] < 0:
        cur[j] = cur[j]
    else:
        r = ratio_theta(j, cur, temp, gamma, neigh, coord, N)    # Hastings ratio
        u = np.random.uniform() # generate uniform r.v.
        cur[j] = temp[j]*(r > u)+cur[j]*(r < u)    # update theta[j]
#     if cur[j] != temp[j]:
#         accept.append(0)
#     else:
#         accept.append(1)
#     with open(dirname+'/accept.txt', 'w') as f_accept:
#         pickle.dump(accept, f_accept) 
    return cur

# update gamma
cpdef np.ndarray[double, ndim = 2] update_gamma(int v, int j, np.ndarray[double, ndim = 2] gamma_cur, np.ndarray[double, ndim = 1] theta_cur, np.ndarray[double, ndim = 2] coord, dict neigh, dict cov_m_inv, np.ndarray[double, ndim = 2] data, double tp, np.ndarray[double, ndim = 2] design_m):
    
    cdef double r, u
    cdef np.ndarray[double, ndim = 2] cur, temp
    
    cur = np.copy(gamma_cur)
    temp = np.copy(gamma_cur)
    temp[v, j] = np.random.binomial(1, gamma_prop(v, j, gamma_cur, theta_cur, coord, neigh))    # proposal walk
    if temp[v, j] != cur[v, j]:
        r = ratio_gamma(v, j, cur, temp, cov_m_inv, data, tp, design_m)    # Hastings ratio
        u = np.random.uniform() # generate uniform r.v.
        cur = temp*(r > u)+cur*(r < u)    # update gamma[v, j]
    return cur
    

#### mcmc update for theta & gamma ####
def mcmc_update(np.ndarray[double, ndim = 2] theta, dict gamma, np.ndarray[double, ndim = 2] coord, dict neigh, dict cov_m_inv, np.ndarray[double, ndim = 2] data, double tp, np.ndarray[double, ndim = 2] design_m, int p, int N, char dirname):
    
    cdef int thresh = 1000   # threshold for checking mcmcse
    cdef int n = 0   # start simulation
    cdef int v, j
    cdef np.ndarray[double, ndim = 2] gamma_cur, comb_cur, comb
    cdef np.ndarray[double, ndim = 1] theta_cur, e, se, ssd
    
    comb = np.append(gamma[0].flatten(), theta.flatten())  # storage of all parameters
    
    while 1:
        n += 1  # counts
        theta_cur = np.copy(theta[n-1, :])  # latest theta
        gamma_cur = np.copy(gamma[n-1])  # laste gamma
        
        # update gamma
        for v from 0 <= v < N:
            for j from 2<= j < p: # first two colums are one's
                gamma_cur = update_gamma(v, j, gamma_cur, theta_cur, coord, neigh, cov_m_inv, data, tp, design_m) # update gamma
        gamma.update({n: gamma_cur})
        # write gamma in file
        with open(dirname+'/gamma.txt', 'w') as f_gamma:
            pickle.dump(gamma, f_gamma)
        
        # update theta
        for j from 0 <= j < p:
            theta_cur = update_theta(j, theta_cur, gamma_cur, neigh, coord, N, dirname)   # update theta
        theta = np.vstack([theta, theta_cur])
        # write theta in file
        with open(dirname+'/theta.txt', 'w') as f_theta:
            pickle.dump(theta, f_theta)
        
        # evaluate mcse
        comb_cur = np.append(gamma_cur, theta_cur)
        comb = np.vstack((comb, comb_cur))  
        if n > thresh:
            thresh += 100
            with open(dirname+'/comb.txt', 'w') as f_comb:
                pickle.dump(comb, f_comb)
            [e, se] = mcse(comb.T)
            ssd = np.std(comb, 0)
            if np.prod(se*1.645+1./n < 0.05*ssd): # 90% and epsilon
                break
                
        
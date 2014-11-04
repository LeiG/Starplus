#!/usr/bin/python

"""
This program conducts the posterior analysis for the fMRI study -
StarPlus. It contains the posterior distributions, the MCMC algorithms
utilized and the stopping criteria used to terminate simulations.
"""


#cython:boundscheck(False)
#cython:wraparound(False)
#defining NPY_NO_DEPRECATED_API NPY_1_7_1

import numpy as np
cimport numpy as np
np.import_array()
import copy
from numpy.linalg import inv
from scipy.special import gamma as Gamma
import pickle
from scipy.stats import norm

#cpdef np.ndarray[double, ndim = 2] mcse(np.ndarray[double, ndim = 2] y):
#    """
#    The mcse() calucates the Monte Carlo standard error.
#
#    Inputs
#    ------
#    y: a dim-by-length matrix records a MCMC simulation
#
#    Returns
#    -------
#    se_hat: a vector records the MC standard error
#    """
#
#
#    cdef unsigned int length = y.shape[1]
#    cdef unsigned int dim = y.shape[0]
#    cdef unsigned int b = np.int(np.sqrt(length))
#    cdef unsigned int a = length//b
#    cdef unsigned int d, k
#    cdef np.ndarray batch = np.zeros([dim, a], dtype = np.float)
#    cdef np.ndarray mu_hat = np.zeros([dim, 1], dtype = np.float)
#    cdef np.ndarray se_hat = np.zeros([dim, 2], dtype = np.float)
#
#    for d in range(dim):
#        for k in range(a):
#            batch[d, k] = np.mean(y[d, (k*b):((k+1)*b)])
#        mu_hat[d] = np.mean(batch[d])
#        se_hat[d, 0] = np.sqrt(b*np.sum((batch[d] - mu_hat[d])**2)/(a - 1))  #mcse
#        se_hat[d, 1] = np.std(y[d])  #standard deviation
#
#    return se_hat[:, 1]


# neighborhood structure ####
def neighbor(int N, np.ndarray[double, ndim = 2] coord):
    """
    Defines the neighborhood structure
    """

    cdef dict output = {} # dict for neighbor
    cdef int v, k
    cdef double dis = 0.0
    cdef np.ndarray[double, ndim = 2] w = np.zeros((N, N))

    for v from 0 <= v < N:
        for k from 0 <= k < v:
            dis = np.sqrt((coord[v][0]-coord[k][0])**2+(coord[v][1]-coord[k][1])**2+(coord[v][2]-coord[k][2])**2)
            if dis == 0.0:
                w[v, k] = w[k, v] = 0.0
            else:
                w[v, k] = w[k, v] = 1.0/dis

    for v from 0 <= v < N:
        output.update({v: np.where(w[v] == 1.0)[0]})

    return output, w


#### MLE for rho and sig ####
# log-likelihood and its derivative and hessian matrix
def loglike_ar1(np.ndarray[double, ndim = 1] x, np.ndarray[double, ndim = 1] y):

    cdef double rho = x[0]
    cdef double sig = x[1]
    cdef int T_int = len(y)
    cdef double T = np.float(T_int)
    cdef double part_1 = 0.0
    cdef double part_2 = 0.0
    cdef double part_3 = 0.0
    cdef np.ndarray[double, ndim = 2] hess = np.zeros((2, 2), dtype = np.float)
    cdef int i
    cdef double loglikelihood, der_rho, der_sig

    for i in range(T_int - 1):
        part_1 += (y[i+1]-rho*y[i])**2.0
        part_2 += (y[i+1] - rho*y[i])*y[i]
        part_3 += y[i]**2.0

    if rho <= -1 or rho >= 1 or sig <= 0:
        loglikelihood = -np.inf
    else:
        loglikelihood = -T*np.log(2.0*np.pi)/2.0-(T-1.0)*np.log(sig*(1.0-rho**2.0))/2.0-part_1/(2.0*sig*(1.0-rho**2.0))-np.log(sig)/2.0-y[0]**2.0/(2.0*sig)   # log-likelihood

    der_rho = (T-1.0)*rho/(1.0-rho**2.0) - rho*part_1/(sig*(1.0-rho**2.0)**2.0) + part_2/(sig*(1.0-rho**2.0))
    der_sig = -(T-1.0)/(2.0*sig) + part_1/(2.0*(1.0-rho**2.0)*(sig**2.0)) - 1.0/(2.0*sig) + y[0]**2.0/(2.0*(sig**2.0))

    hess[0, 0] = (T-1.0)*(1.0+rho**2.0)/((1.0-rho**2.0)**2.0) - (1.0+3.0*rho**2.0)*part_1/(sig*(1.0-rho**2.0)**3.0) + 4.0*rho*part_2/(sig*(1.0-rho**2.0)**2.0) - part_3/(sig*(1.0-rho**2.0))
    hess[0, 1] = rho*part_1/(((1.0-rho**2.0)**2.0)*(sig**2.0)) - part_2/((sig**2.0)*(1.0-rho**2.0))
    hess[1, 0] = hess[0, 1]
    hess[1, 1] = (T-1.0)/(2.0*sig**2.0) - part_1/((1.0-rho**2.0)*(sig**3.0)) + 1.0/(2.0*sig**2.0) - y[0]**2.0/(sig**3.0)

    return [loglikelihood, [der_rho, der_sig], hess]

# Newton function
def Newton(np.ndarray[double, ndim = 1] x_0, np.ndarray[double, ndim = 1] data):

    cdef double eps = 0.000001
    cdef list f_0 = loglike_ar1(x_0, data)
    cdef np.ndarray[double, ndim = 1] y_new = x_0 - np.dot(inv(f_0[2]), f_0[1])
    cdef list f_new = loglike_ar1(y_new, data)
    while np.fabs(f_new[0] - f_0[0]) > eps:
        f_0 = copy.deepcopy(f_new)
        y_new -= np.dot(inv(f_0[2]), f_0[1])
        f_new = loglike_ar1(y_new, data)

    return y_new

# MLE for rho and sigma
def rhosig_mle(np.ndarray[double, ndim = 2] data, int N):

    cdef int v
    cdef np.ndarray output = np.zeros((N, 2), dtype = np.float)
    cdef np.ndarray x_0 = np.array([0.0, 1.0])

    for v from 0 <= v < N:
        output[v, :] = Newton(x_0, data[:, v])

    return output


#### covariance matrix ####
def cov_matrix(np.ndarray[double, ndim = 1] rho, int N, int tp):

    cdef int v, i, j
    cdef np.ndarray cov = np.zeros([tp, tp], dtype = np.float)
    cdef np.ndarray output = np.zeros([N, tp, tp], dtype = np.float)
    cdef np.ndarray output_inv = np.zeros([N, tp, tp], dtype = np.float)

    for v in range(N):
        for i in range(tp):
            for j from 0 <= j < i:
                cov[i, j] = cov[j, i] = rho[v]**np.fabs(i - j)
        output[v] = cov
        output_inv[v] = inv(cov)

    return output, output_inv


#### design matrix ####
# convolve function
def conv(np.ndarray[double, ndim = 1] hrf, np.ndarray[double, ndim = 1] sti):

    cdef np.ndarray[double, ndim = 1] output = np.arange(np.float(np.size(sti)))
    cdef int value, i

    for value in output:
        output[value] = 0.0
        for i in range(value+1):
            output[value] += hrf[i]*sti[value - i]
    return output

cpdef np.ndarray[double, ndim = 2] design(int tp, double press):

    cdef np.ndarray sti_0 = np.zeros(tp, dtype = np.float)
    cdef np.ndarray sti_1 = np.zeros(tp, dtype = np.float)
    cdef np.ndarray sti_2 = np.zeros(tp, dtype = np.float)
    cdef np.ndarray hrf = np.zeros(tp, dtype = np.float)
    cdef np.ndarray t = np.arange(0, np.float(tp)/2.0, 0.5)
    cdef int tt
    cdef double t_press = min(12.0, press)
    cdef np.ndarray output = np.zeros((tp, 4), dtype = np.float)

    for tt in range(tp):
        if t[tt] == 0.0:
            sti_0[tt] = 1.0
        else:
            sti_0[tt] = 0.0

        if t[tt] <= 4.0 and t[tt] >= 0.0:
            sti_1[tt] = 1.0
        else:
            sti_1[tt] = 0.0

        if t[tt] >= 8.0 and t[tt] <= t_press:
            sti_2[tt] = 1.0
        else:
            sti_2[tt] = 0.0

        hrf[tt] = 5.6999155101700625*((t[tt]**5)*np.exp(-t[tt])/Gamma(6.0)-1/6.0*(t[tt]**15)*np.exp(-t[tt])/Gamma(16.0))/9.5187445708326752

    output = np.ones((tp, 4))
    output[:, 1] = conv(hrf, sti_0)
    output[:, 2] = conv(hrf, sti_1)
    output[:, 3] = conv(hrf, sti_2)
    return output


#### Ising prior ####
cpdef double log_Ising(double theta, np.ndarray[double, ndim = 1] gamma, dict neigh, int N, double a):

    cdef double output = 0.0
    cdef int v, k

    for v in range(N):
        output += a*gamma[v]
        for k in neigh[v]:
            if gamma[k] == gamma[v]:
                output += theta # subject to neighborhood structure/ weight changes

    return output


#### MCMC components ####
# S function
cpdef np.ndarray[double, ndim = 2] S(int v, np.ndarray[double, ndim = 2] cov_inv, np.ndarray[double, ndim = 2] gamma, np.ndarray[double, ndim = 1] data, double tp, np.ndarray[double, ndim = 2] design):

    cdef np.ndarray[double, ndim = 2] d_nonzero, beta, output, temp1, temp2
    cdef np.ndarray[double, ndim = 2] y = data.reshape(tp, 1)

    d_nonzero = design[:,np.nonzero(gamma[v,:])[0]] # design matrix nonzero part
    temp1 = d_nonzero.T.dot(cov_inv)
    beta = inv(temp1.dot(d_nonzero)).dot(temp1.dot(y))
    temp2 = y-d_nonzero.dot(beta)
    output = (temp2.T.dot(cov_inv)).dot(temp2)
    return output

# update gamma
cpdef double update_gamma(int v, int j, np.ndarray[double, ndim = 2] gamma_cur, double theta_cur, np.ndarray[long, ndim = 1] neigh, np.ndarray[double, ndim = 2] cov_m_inv, np.ndarray[double, ndim = 1] data, double tp, np.ndarray[double, ndim = 2] design_m):

    cdef np.ndarray[double, ndim = 2] cur = np.copy(gamma_cur)
    cdef np.ndarray[double, ndim = 2] temp = np.copy(gamma_cur)
    cdef double prop_part = -0.1*cur[v, j]
    cdef double r, u, S1, S2
    cdef int k

    for k in neigh:
        prop_part += theta_cur*(1.0-2.0*cur[k, j])

    temp[v, j] = np.random.binomial(1, (1.0/(1.0+np.exp(prop_part))))
    if temp[v, j] != cur[v, j]:
        S1 = S(v, cov_m_inv, temp, data, tp, design_m)
        S2 = S(v, cov_m_inv, cur, data, tp, design_m)
        r = (1+tp)**(-temp[v, j]/2+cur[v, j]/2)*(S1/S2)**(tp/2)
        u = np.random.uniform() # generate uniform r.v.
        if r > u:
            cur[v, j] = temp[v, j]    # update gamma[v, j]

    return cur[v, j]

# log ratio of normalizing constant by path sampling
cpdef double log_const_ratio(int j, np.ndarray[double, ndim = 1] cur, np.ndarray[double, ndim = 1] temp, np.ndarray[double, ndim = 2] gamma_cur, dict neigh, int N, np.ndarray[double, ndim = 3] cov_m_inv, np.ndarray[double, ndim = 2] data, double tp, np.ndarray[double, ndim = 2] design_m):

    cdef double low = min(cur[j], temp[j])
    cdef double high = max(cur[j], temp[j])
    cdef double highlow = high - low
    cdef np.ndarray[double, ndim = 2] gamma_path = np.copy(gamma_cur)
    cdef np.ndarray[double, ndim = 1] theta_path = np.copy(temp)
    cdef unsigned int iter = 0
    cdef int v, k
    cdef double output
#     cdef double prop_part = 0.0
    cdef np.ndarray[double, ndim = 1] sample = np.array(log_Ising(theta_path[j], gamma_path[:, j+2], neigh, N, 0.1)/highlow, ndmin = 1)

    while iter < 100000:
        iter += 1   # iterations
        theta_path[j] = np.random.uniform(low, high)    # generate transitional theta
        for v in range(N):
            # prop_part = -0.1*gamma_path[v, j]
#             for k in neigh[v]:
#                 prop_part += theta_path*(1.0-2.0*gamma_path[k, j])
#             gamma_path[v, j] = np.random.binomial(1, (1.0/(1.0+np.exp(prop_part))))
            gamma_path[v, j+2] = update_gamma(v, j+2, gamma_path, theta_path[j], neigh[v], cov_m_inv[v], data[:, v], tp, design_m)

        sample = np.append(sample, (log_Ising(theta_path[j], gamma_path[:, j+2], neigh, N, 0.0)/highlow))

    output = np.average(sample)
#     if cur[j] > temp[j]:
#         return output
#     else:
#         return -output
    return output


# update theta
cpdef double update_theta(int j, np.ndarray[double, ndim = 1] theta_cur, np.ndarray[double, ndim = 2] gamma_cur, double log_const, dict neigh, int N):

    cdef double theta_max = 1.0   # theta_max
    cdef np.ndarray[double, ndim = 1] cur = np.copy(theta_cur)
    cdef np.ndarray[double, ndim = 1] temp = np.copy(theta_cur)
    cdef double log_r, r, u

    temp[j] = np.random.normal(cur[j], 1)  # generate proposal r.v.
    if cur[j] > theta_max or cur[j] < 0.0:
        cur[j] = temp[j]
    elif temp[j] > theta_max or temp[j] < 0.0:
        cur[j] = cur[j]
    else:
        log_r = log_const+log_Ising(temp[j]-cur[j], gamma_cur[:, j+2], neigh, N, 0.0)+np.log(norm.pdf(cur[j], cur[j], 1)/norm.pdf(temp[j], cur[j], 1))
#         log_r = log_Ising(temp[j]-cur[j], gamma_cur[:, j], neigh, N, 0.0)+np.log(norm.pdf(cur[j], cur[j], 0.6)/norm.pdf(temp[j], temp[j], 0.6))
        if log_r > 0.0:
            r = 1.0
        else:
            r = np.exp(log_r)
        u = np.random.uniform() # generate uniform r.v.
        if r > u:
            cur[j] = temp[j]    # update theta[j]

    return cur[j]


#### MCMC updates ####
cpdef int mcmc_update(dict neigh, np.ndarray[double, ndim = 3] cov_m_inv, np.ndarray[double, ndim = 2] data, double tp, np.ndarray[double, ndim = 2] design_m, int p, int N, bytes dirname):

    cdef int thresh = 0  # threshold (in batches) for checking mcmcse
    cdef unsigned int n = 1   # start simulation
    cdef int b_n = 1   # batch counts
    cdef int i, v, j
    cdef np.ndarray[long, ndim = 1] b = (2**7)*np.ones(2, dtype = np.int)   # start to check at 2**(2*7)
    cdef double a
    cdef np.ndarray[double, ndim = 2] gamma_cur, gamma, theta, gamma_test, gamma_std, theta_std, gamma_batch, theta_batch
    cdef np.ndarray[double, ndim = 1] theta_cur, theta_test, cond_theta, cond_gamma, b_array, mcse_theta, mcse_gamma
    cdef np.ndarray[double, ndim = 1] log_const = np.zeros(p-2)


    b_array = np.ones(6)
    for i in range(6):
        b_array[i] = 2**(7+i)

    # initial values
#     theta = np.random.uniform(0.0, 1.0, size = (1, p-2)) # strength of interaction
#     theta_cur = np.copy(theta[0])
    theta_cur = np.random.uniform(0.0, 1.0, size = p-2)
    gamma_cur = 1.0*np.random.randint(2, size = (N, p))
    gamma_cur[:, 0:2] = 1.0
#     gamma = np.array(gamma_cur[:, 2:p].T.flatten(), ndmin = 2)   # indicator gamma

    gamma_batch = np.empty((b[1], N*(p-2)))
    theta_batch = np.empty((b[1], p-2))

    gamma_batch[0] = gamma_cur[:, 2:p].T.flatten()
    theta_batch[0] = theta_cur

    theta_test = np.random.uniform(0.0, 1.0, size = p-2)
    gamma_test = 1.0*np.random.randint(2, size = (N, p))
    gamma_test[:, 0:2] = 1.0

    # pre-run ratio of normalizing constant
    for j in range(p-2):
        log_const[j] = log_const_ratio(j, theta_cur, theta_test, gamma_test, neigh, N, cov_m_inv, data, tp, design_m)

    with open(dirname+'/logconst.txt', 'w') as f_log_const:
        f_log_const.write(str(log_const))

    gamma_std = np.zeros((N*(p-2), 3))
    theta_std = np.zeros((p-2, 3))

    theta_std[:, 0] = theta_batch[0]
    gamma_std[:, 0] = gamma_batch[0]

#     f_accept = open(dirname+'/accept.txt', 'w')
#     f_accept.close()
#     f_an = open(dirname+'/an.txt', 'w')
#     f_an.close()

    while 1:
        n += 1  # counts

        # update gamma
        for j in range(2, p):
            for v in range(N):
                gamma_cur[v, j] = update_gamma(v, j, gamma_cur, theta_cur[j-2], neigh[v], cov_m_inv[v], data[:, v], tp, design_m) # update gamma
#         gamma = np.vstack([gamma, gamma_cur[:, 2:p].T.flatten()])
        gamma_batch[b_n] = gamma_cur[:, 2:p].T.flatten()

        # update theta
        for j in range(p-2):
#             temp = theta_cur[j]
            theta_cur[j] = update_theta(j, theta_cur, gamma_cur, log_const[j], neigh, N)   # update theta
#             if temp == theta_cur[j]:
#                 with open(dirname+'/accept.txt', 'a') as f_accept:
#                     f_accept.write(str(0))
#             else:
#                 with open(dirname+'/accept.txt', 'a') as f_accept:
#                     f_accept.write(str(1))
#         theta = np.vstack([theta, theta_cur])
        theta_batch[b_n] = theta_cur


        # update \bar{x_n}
        theta_std[:, 1] = ((n-1)*theta_std[:, 0]+theta_batch[b_n])/n
        gamma_std[:, 1] = ((n-1)*gamma_std[:, 0]+gamma_batch[b_n])/n

        # update \bar{\sigma_n^2}
        theta_std[:, 2] = theta_std[:, 2]+n*(theta_std[:, 1]-theta_batch[b_n])**2/(n-1)
        gamma_std[:, 2] = gamma_std[:, 2]+n*(gamma_std[:, 1]-gamma_batch[b_n])**2/(n-1)

        theta_std[:, 0] = theta_std[:, 1]
        gamma_std[:, 0] = gamma_std[:, 1]


        b_n += 1    # update batch count

        if b_n == b[1]:
            b_n = 0
            if thresh == 0:
                gamma = np.array(np.average(gamma_batch, 0), ndmin = 2)
                theta = np.array(np.average(theta_batch, 0), ndmin = 2)
                thresh = 1
            else:
                gamma = np.vstack([gamma, np.average(gamma_batch, 0)])
                theta = np.vstack([theta, np.average(theta_batch, 0)])
                thresh += 1


        # check every 20 or 21 batch
        if n >= 2**(2*7) and thresh > 20 and gamma.shape[0]%2 == 0:

            np.savetxt(dirname+'/n.txt', [n])

            thresh = 1

            np.save(dirname+'/gamma', gamma)
            np.save(dirname+'/theta', theta)

            np.save(dirname+'/gamma_var', gamma_std[:, 2])
            np.save(dirname+'/theta_var', theta_std[:, 2])


            b[0] = b[1]
            b[1] = 2**(max(np.where(b_array <= np.sqrt(n))[0])+7)

            # merge if batch size change
            if b[0] != b[1]:
                gamma_batch = np.empty((b[1], N*(p-2)))
                theta_batch = np.empty((b[1], p-2))
                gamma = np.average(np.array(np.vsplit(gamma, gamma.shape[0]/2)), 1)
                theta = np.average(np.array(np.vsplit(theta, theta.shape[0]/2)), 1)

            # calculate mcse
            a = n/b[1]
#             np.savetxt(dirname+'/a.txt', [a])
#             np.savetxt(dirname+'/size.txt', [gamma.shape[0]])
#             with open(dirname+'/an.txt', 'a') as f_an:
#                 np.savetxt(f_an, [a, b[1]])

            mcse_gamma = np.sqrt((np.sum((gamma - np.average(gamma, 0))**2, 0)*b[1]/(a-1))/n)
            mcse_theta = np.sqrt((np.sum((theta - np.average(theta, 0))**2, 0)*b[1]/(a-1))/n)

#             with open(dirname+'/n.txt', 'w') as f_n:
#                 f_n.write(str(n))

#             np.savetxt(dirname+'/n.txt', n)

            # write gamma in file
#             with open(dirname+'/gamma.txt', 'w') as f_gamma:
#                 pickle.dump(gamma, f_gamma)
#             # write theta in file
#             with open(dirname+'/theta.txt', 'w') as f_theta:
#                 pickle.dump(theta, f_theta)

#             mcse_theta = mcse(theta.T)
#             mcse_gamma = mcse(gamma.T)

#             cond_theta = (mcse_theta[:, 0]*1.645+1.0/n - 0.1*mcse_theta[:, 1])
#             cond_gamma = (mcse_gamma[:, 0]*1.645+1.0/n - 0.1*mcse_gamma[:, 1])

            # ESS 4000 -> epsilon 0.062
            cond_theta = (2*mcse_theta*1.96 - 0.062*np.sqrt(theta_std[:, 2]/(n-1)))
            cond_gamma = (2*mcse_gamma*1.96 - 0.062*np.sqrt(gamma_std[:, 2]/(n-1)))

            np.savetxt(dirname+'/cond_theta.txt', cond_theta, delimiter=',')
            np.savetxt(dirname+'/cond_gamma.txt', cond_gamma, delimiter=',')

            if np.all(cond_theta <= 0) and np.all(cond_gamma <= 0):
                break
#             if np.prod(se*1.645+1.0/n < 0.1*ssd): # 90% and epsilon = 0.05
#                 break

    with open(dirname+'/done.txt', 'w') as f_done:
        f_done.write('done')

    return 0


#### MCMC updates with ESS ####
cpdef int mcmc_ess_update(dict neigh, np.ndarray[double, ndim = 3] cov_m_inv, np.ndarray[double, ndim = 2] data, double tp, np.ndarray[double, ndim = 2] design_m, int p, int N, bytes dirname):

    cdef int thresh = 0  # threshold (in batches) for checking mcmcse
    cdef unsigned int n = 1   # start simulation
    cdef int b_n = 1   # batch counts
    cdef int i, v, j
    cdef np.ndarray[long, ndim = 1] b = (2**7)*np.ones(2, dtype = np.int)   # start to check at 2**(2*7)
    cdef double a
    cdef np.ndarray[double, ndim = 2] gamma_cur, gamma, theta, gamma_test, gamma_std, theta_std, gamma_batch, theta_batch
    cdef np.ndarray[double, ndim = 1] theta_cur, theta_test, cond_theta, cond_gamma, b_array, mcse_theta, mcse_gamma
    cdef np.ndarray[double, ndim = 1] log_const = np.zeros(p-2)


    b_array = np.ones(6)
    for i in range(6):
        b_array[i] = 2**(7+i)

    # initial values
#     theta = np.random.uniform(0.0, 1.0, size = (1, p-2)) # strength of interaction
#     theta_cur = np.copy(theta[0])
    theta_cur = np.random.uniform(0.0, 1.0, size = p-2)
    gamma_cur = 1.0*np.random.randint(2, size = (N, p))
    gamma_cur[:, 0:2] = 1.0
#     gamma = np.array(gamma_cur[:, 2:p].T.flatten(), ndmin = 2)   # indicator gamma

    gamma_batch = np.empty((b[1], N*(p-2)))
    theta_batch = np.empty((b[1], p-2))

    gamma_batch[0] = gamma_cur[:, 2:p].T.flatten()
    theta_batch[0] = theta_cur

    theta_test = np.random.uniform(0.0, 1.0, size = p-2)
    gamma_test = 1.0*np.random.randint(2, size = (N, p))
    gamma_test[:, 0:2] = 1.0

    # pre-run ratio of normalizing constant
    for j in range(p-2):
        log_const[j] = log_const_ratio(j, theta_cur, theta_test, gamma_test, neigh, N, cov_m_inv, data, tp, design_m)

    with open(dirname+'/logconst.txt', 'w') as f_log_const:
        f_log_const.write(str(log_const))

    gamma_std = np.zeros((N*(p-2), 3))
    theta_std = np.zeros((p-2, 3))

    theta_std[:, 0] = theta_batch[0]
    gamma_std[:, 0] = gamma_batch[0]

#     f_accept = open(dirname+'/accept.txt', 'w')
#     f_accept.close()
#     f_an = open(dirname+'/an.txt', 'w')
#     f_an.close()

    while 1:
        n += 1  # counts

        # update gamma
        for j in range(2, p):
            for v in range(N):
                gamma_cur[v, j] = update_gamma(v, j, gamma_cur, theta_cur[j-2], neigh[v], cov_m_inv[v], data[:, v], tp, design_m) # update gamma
#         gamma = np.vstack([gamma, gamma_cur[:, 2:p].T.flatten()])
        gamma_batch[b_n] = gamma_cur[:, 2:p].T.flatten()

        # update theta
        for j in range(p-2):
#             temp = theta_cur[j]
            theta_cur[j] = update_theta(j, theta_cur, gamma_cur, log_const[j], neigh, N)   # update theta
#             if temp == theta_cur[j]:
#                 with open(dirname+'/accept.txt', 'a') as f_accept:
#                     f_accept.write(str(0))
#             else:
#                 with open(dirname+'/accept.txt', 'a') as f_accept:
#                     f_accept.write(str(1))
#         theta = np.vstack([theta, theta_cur])
        theta_batch[b_n] = theta_cur


        # update \bar{x_n}
        theta_std[:, 1] = ((n-1)*theta_std[:, 0]+theta_batch[b_n])/n
        gamma_std[:, 1] = ((n-1)*gamma_std[:, 0]+gamma_batch[b_n])/n

        # update \bar{\sigma_n^2}
        theta_std[:, 2] = theta_std[:, 2]+n*(theta_std[:, 1]-theta_batch[b_n])**2/(n-1)
        gamma_std[:, 2] = gamma_std[:, 2]+n*(gamma_std[:, 1]-gamma_batch[b_n])**2/(n-1)

        theta_std[:, 0] = theta_std[:, 1]
        gamma_std[:, 0] = gamma_std[:, 1]


        b_n += 1    # update batch count

        if b_n == b[1]:
            b_n = 0
            if thresh == 0:
                gamma = np.array(np.average(gamma_batch, 0), ndmin = 2)
                theta = np.array(np.average(theta_batch, 0), ndmin = 2)
                thresh = 1
            else:
                gamma = np.vstack([gamma, np.average(gamma_batch, 0)])
                theta = np.vstack([theta, np.average(theta_batch, 0)])
                thresh += 1


        # check every 20 or 21 batch
        if n >= 2**(2*7) and thresh > 20 and gamma.shape[0]%2 == 0:

            np.savetxt(dirname+'/n.txt', [n])

            thresh = 1

            np.save(dirname+'/gamma', gamma)
            np.save(dirname+'/theta', theta)

            np.save(dirname+'/gamma_var', gamma_std[:, 2])
            np.save(dirname+'/theta_var', theta_std[:, 2])


            b[0] = b[1]
#             b[1] = 2**(max(np.where(b_array <= np.sqrt(n))[0])+7) #lower bound
            b[1] = 2**(min(np.where(b_array >= np.sqrt(n))[0])+7) #upper bound

            # merge if batch size change
            if b[0] != b[1]:
                gamma_batch = np.empty((b[1], N*(p-2)))
                theta_batch = np.empty((b[1], p-2))
                gamma = np.average(np.array(np.vsplit(gamma, gamma.shape[0]/2)), 1)
                theta = np.average(np.array(np.vsplit(theta, theta.shape[0]/2)), 1)

            # calculate mcse
            a = n/b[1]
#             np.savetxt(dirname+'/a.txt', [a])
#             np.savetxt(dirname+'/size.txt', [gamma.shape[0]])
#             with open(dirname+'/an.txt', 'a') as f_an:
#                 np.savetxt(f_an, [a, b[1]])

            mcse_gamma = np.sqrt((np.sum((gamma - np.average(gamma, 0))**2, 0)*b[1]/(a-1))/n)
            mcse_theta = np.sqrt((np.sum((theta - np.average(theta, 0))**2, 0)*b[1]/(a-1))/n)

#             with open(dirname+'/n.txt', 'w') as f_n:
#                 f_n.write(str(n))

#             np.savetxt(dirname+'/n.txt', n)

            # write gamma in file
#             with open(dirname+'/gamma.txt', 'w') as f_gamma:
#                 pickle.dump(gamma, f_gamma)
#             # write theta in file
#             with open(dirname+'/theta.txt', 'w') as f_theta:
#                 pickle.dump(theta, f_theta)

#             mcse_theta = mcse(theta.T)
#             mcse_gamma = mcse(gamma.T)

#             cond_theta = (mcse_theta[:, 0]*1.645+1.0/n - 0.1*mcse_theta[:, 1])
#             cond_gamma = (mcse_gamma[:, 0]*1.645+1.0/n - 0.1*mcse_gamma[:, 1])

#             cond_theta = (2*mcse_theta*1.96 - 0.1*np.sqrt(theta_std[:, 2]/(n-1)))
#             cond_gamma = (2*mcse_gamma*1.96 - 0.1*np.sqrt(gamma_std[:, 2]/(n-1)))
            cond_theta = (theta_std[:, 2]/(n-1))/(mcse_theta**2)
            cond_gamma = (gamma_std[:, 2]/(n-1))/(mcse_gamma**2)

            np.savetxt(dirname+'/cond_theta.txt', cond_theta, delimiter=',')
            np.savetxt(dirname+'/cond_gamma.txt', cond_gamma, delimiter=',')

            if np.all(cond_theta > 4000) and np.all(cond_gamma > 4000):
                break
#             if np.prod(se*1.645+1.0/n < 0.1*ssd): # 90% and epsilon = 0.05
#                 break

    with open(dirname+'/done.txt', 'w') as f_done:
        f_done.write('done')

    return 0


#### MCMC convergence diagnostic ####
cpdef int mcmc_diag(dict neigh, np.ndarray[double, ndim = 3] cov_m_inv, np.ndarray[double, ndim = 2] data, double tp, np.ndarray[double, ndim = 2] design_m, int p, int N, bytes dirname):

    cdef int thresh = 50000  # threshold (in batches) for checking mcmcse
    cdef int burnin = 5000  # burn-in period
    cdef unsigned int n = 1   # start simulation
    cdef int i, v, j
    cdef np.ndarray[double, ndim = 2] gamma_cur, gamma, theta, gamma_test
    cdef np.ndarray[double, ndim = 1] theta_cur, theta_test
    cdef np.ndarray[double, ndim = 1] log_const = np.zeros(p-2)


    # container
    gamma = np.empty((thresh, N*(p-2)))
    theta = np.empty((thresh, p-2))

    # initial values
    theta_cur = np.random.uniform(0.0, 1.0, size = p-2)
    gamma_cur = 1.0*np.random.randint(2, size = (N, p))
    gamma_cur[:, 0:2] = 1.0

    gamma[0] = gamma_cur[:, 2:p].T.flatten()
    theta[0] = theta_cur

    theta_test = np.random.uniform(0.0, 1.0, size = p-2)
    gamma_test = 1.0*np.random.randint(2, size = (N, p))
    gamma_test[:, 0:2] = 1.0

    # pre-run ratio of normalizing constant
    for j in range(p-2):
        log_const[j] = log_const_ratio(j, theta_cur, theta_test, gamma_test, neigh, N, cov_m_inv, data, tp, design_m)

    with open(dirname+'/logconst.txt', 'w') as f_log_const:
        f_log_const.write(str(log_const))


    while n < (thresh-1):
        n += 1  # counts

        # update gamma
        for j in range(2, p):
            for v in range(N):
                gamma_cur[v, j] = update_gamma(v, j, gamma_cur, theta_cur[j-2], neigh[v], cov_m_inv[v], data[:, v], tp, design_m) # update gamma
        gamma[n] = gamma_cur[:, 2:p].T.flatten()

        # update theta
        for j in range(p-2):
            theta_cur[j] = update_theta(j, theta_cur, gamma_cur, log_const[j], neigh, N)   # update theta
        theta[n] = theta_cur

    np.savetxt(dirname+'/gamma.txt', gamma, delimiter=',')
    np.savetxt(dirname+'/theta.txt', theta, delimiter=',')

    # discard burn-in period
    gamma = gamma[burnin:, :]
    theta = theta[burnin:, :]

    np.savetxt(dirname+'/gamma_burnin.txt', gamma, delimiter=',')
    np.savetxt(dirname+'/theta_burnin.txt', theta, delimiter=',')


    with open(dirname+'/done.txt', 'w') as f_done:
        f_done.write('done')

    return 0

#!/usr/bin/python


# import modules
import sys
sys.path.append('/rhome/lgong/mcmcse')
import mcmcse

import pickle
import numpy as np
import time
import os
from scipy import r_
from scipy.io import loadmat
from numpy.linalg import inv
from math import log, exp, sqrt, gamma, pi, fabs


# make new directory to store results
dirfmt = "%4d-%02d-%02d %02d:%02d"
current_time = time.localtime()[0:5]
dirname = dirfmt % current_time
os.mkdir(dirname)   # make new directory


# set random seed
np.random.seed(3)


# functions
def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def sti_0(t):   # initial stimulus
    if t == 0:
        return 1.0
    else:
        return 0.0
        
sti_0 = np.vectorize(sti_0) # vectorize

def sti_1(t):  # first stimulus
    if t <= 4 and t >= 0:
        return 1.0
    if t > 4 and t <= 27:
        return 0.0
        
sti_1 = np.vectorize(sti_1)   # vectorize
        
def sti_2(t, press):  # second stimulus
    t_press = min(12, press)
    if t > 8 and t <= t_press:
        return 1.0
    if t <= 8 and t >= 0:
        return 0.0
    if t > t_press and t <= 27:
        return 0.0
        
sti_2 = np.vectorize(sti_2)   # vectorize

def hrf(t): # HRF function scaled such that sum(hrf(t))=1 for t = np.arange(0, 27, 0.5)
    if t < 0:
        return 0.0
    else:
        #return 1.0/111.8*(t**8.60)*np.exp(-t/0.547)
        return 5.6999155101700625*((t**5)*np.exp(-t)/gamma(6.0)-1/6.0*(t**15)*np.exp(-t)/gamma(16.0))/9.5187445708326752
                
hrf = np.vectorize(hrf) #vectorize

def conv(hrf, sti):  # convolved impulse
    output = range(np.size(sti))
    for value in output:
        output[value] = 0
        for i in xrange(value+1):
            output[value] += hrf[i]*sti[value - i]
    return output

def design(t, press):  # time dependent design matrix
    output = np.ones((np.size(t), 4))
    output[:, 1] = conv(hrf(t), sti_0(t))
    output[:, 2] = conv(hrf(t), sti_1(t))
    output[:, 3] = conv(hrf(t), sti_2(t, press))
    return output
    
def cov(t, rho):    # covariance matrix capital gamma
    output = [[rho**fabs(i - j) for i in xrange(np.size(t))] for j in xrange(np.size(t))]
    output = np.array(output)
    return output
    
def w(v, k):    # weight of interaction between voxels
    dis = sqrt((coord[v][0]-coord[k][0])**2+(coord[v][1]-coord[k][1])**2+(coord[v][2]-coord[k][2])**2)  # distance between voxels
    if dis == 0.:
        return 0.
    else:
        return 1/dis    # weight set to the reciprocal of distance
    
w = np.vectorize(w) # vectorize
    
def neig(v):   # find neighbors of voxel v
    output = [i for i in xrange(N) if w(v, i) == 1.]
    return output
    
def S(v, design, ga, cov): # set S(v, design, $\gamma_v$, $\Gamma_v$) function
    d_nonzero = design[:,np.nonzero(ga[v,:])[0]] # design matrix nonzero part
    cov_inv = inv(cov)  # inverse of covariance matrix
    y = data[:, v].reshape(tp, 1)   # response vector
    beta_head = inv((d_nonzero.T.dot(cov_inv)).dot(d_nonzero)).dot((d_nonzero.T.dot(cov_inv)).dot(y))   # beta estimate given gamma
    output = ((y-d_nonzero.dot(beta_head)).T.dot(cov_inv)).dot(y-d_nonzero.dot(beta_head))  # S function
    return output
    
def log_Ising(j, a, theta, ga):    # log of Ising prior
    s = 0   # second part in Ising prior
    a_T = 0 # first part in Ising prior
    for v in xrange(N):
        for k in neigh[v]:
            if ga[k, j] == ga[v, j]:
                s += w(v, k)
        if roi[v] in G: # information based on anatomical region
            a_T += a*ga[v, j]
    return a_T + theta[j]*s

# for sample correlation estimation of rho    
# def sig_y(rho, y):  # sigma^2 of y
#     y_delay = np.zeros(len(y))  # delayed sequence
#     y_delay[1:] = y[:-1]
#     eps = y - rho*y_delay   # white noise terms
#     sig = np.var(eps)   # variance
#     return sig/(1-rho**2)

def loglike_ar(x, y):   # log likelihood function for AR(1) model
    rho = x[0]
    sig = x[1]
    if rho <= -1 or rho >= 1 or sig <= 0:
        return -inf
    else:
        T = len(y)  # number of time point
        part = 0
        for i in xrange(T-1):
            part += (y[i+1] - rho*y[i])**2 
        return -T*log(2*pi)/2-(T-1)*log(sig*(1-rho**2))/2-part/(2*sig*(1-rho**2))-log(sig)/2-y[0]**2/(2*sig)   # log-likelihood

def loglike_ar_der(x, y):   # derivative of log likelihood for AR(1) model
    rho = x[0]
    sig = x[1]
    T = len(y)  # number of time point
    der_rho = []
    der_sig = []
    part_1 = 0
    part_2 = 0
    for i in xrange(T-1):
        part_1 += (y[i+1] - rho*y[i])**2
        part_2 += (y[i+1] - rho*y[i])*y[i]
    der_rho = (T-1)*rho/(1-rho**2) - rho*part_1/(sig*(1-rho**2)**2) + part_2/(sig*(1-rho**2))
    der_sig = -(T-1)/(2*sig) + part_1/(2*(1-rho**2)*(sig**2)) - 1/(2*sig) + y[0]**2/(2*(sig**2))
    return [der_rho, der_sig]

def loglike_ar_hess(x, y):  # hessian matrix of log likelihood for AR(1) model
    rho = x[0]
    sig = x[1]
    T = len(y)  # number of time point
    hess = np.zeros((2, 2))
    part_1 = 0
    part_2 = 0
    part_3 = 0
    for i in xrange(T-1):
        part_1 += (y[i+1]-rho*y[i])**2
        part_2 += (y[i+1]-rho*y[i])*y[i]
        part_3 += y[i]**2
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

# golden region search method for rho    
# def goldensearch(a, b, f, data):   # golden search algorithm for maximum
#     tau = 10**(-5)  # termination condition
#     phi = (sqrt(5)-1)/2 # golden ratio
#     phi2 = phi**2
#     h = b - a
#     c = a + h*phi2
#     d = a + h*phi
#     fa = -f([a, sig_y(a, data)], data)    # log likelihood
#     fb = -f([b, sig_y(b, data)], data)
#     fc = -f([c, sig_y(c, data)], data)
#     fd = -f([d, sig_y(d, data)], data)
#     while abs(fd - fc) > tau:
#         if fc < fd:
#             b = d
#             d = c
#             h = b - a
#             c = a + phi2*h
#             fb = fd
#             fd = fc
#             fc = -f([c, sig_y(c, data)], data)
#         else:
#             a = c
#             c = d
#             h = b - a
#             d = a + phi*h
#             fa = fc
#             fc = fd
#             fd = -f([d, sig_y(d, data)], data)
#     return (d+c)/2.
    
def ga_prop(v, j, a, ga, theta):   # proposal probability for updating gamma
    part = 0
    for k in neigh[v]:  # second part in proposal distribution
        part += w(v, k)*(1-2*ga[k, j])
    return 1/(1 + exp(-a + theta[j]*part))
    
def ratio_ga(ga_star, ga, S_1, S_2): # Hastings ratio for updating gamma
    return (1+tp)**(-ga_star/2)*S_1**(tp/2)/((1+tp)**(-ga/2)*S_2**(tp/2))
    
# def ratio_rho(rho_star, rho):   # Hastings ratio for updating rho
#     return

# bridge sampling for normalizing constant
# def log_const_theta(j, theta, theta_star, a, ga):  # log normalizing constant in Hastings ratio for updating theta
#     # bridge sampling with alpha = (q_1*q_2)^(-1/2)
#     nsample = 1 # number of samples
#     ga_1 = np.copy(ga)
#     ga_2 = np.copy(ga)
#     r_1 = []    # log r estimate nominator
#     r_2 = []    # log r estimate denominator
#     for i in xrange(nsample):
#         for v in xrange(N):
#             ga_1[v, j] = np.random.binomial(1, ga_prop(v, j, a, ga_1, theta))
#             ga_2[v, j] = np.random.binomial(1, ga_prop(v, j, a, ga_2, theta_star))
#         r_1.append(0.5*(log_Ising(j, a, theta, ga_2) - log_Ising(j, a, theta_star, ga_2)))
#         r_2.append(0.5*(log_Ising(j, a, theta_star, ga_1) - log_Ising(j, a, theta, ga_1)))
#     if max(r_1) > 0:
#         A_1 = max(r_1)
#     else:
#         A_1 = -min(r_1)
#     if max(r_2) > 0:
#         A_2 = max(r_2)
#     else:
#         A_2 = -min(r_2)
#     r_1 = np.array(r_1) - A_1
#     r_2 = np.array(r_2) - A_2
#     r_1 = sum([exp(x) for x in r_1])       
#     r_2 = sum([exp(x) for x in r_2])
#     output = A_1+r_1 - (A_2+r_2)
#     return output
    
# path sampling for normalizing constant
# def log_const_theta(j, theta, theta_star, a, ga):   # log normalizing constant in Hastings ratio for updating theta
#     # path sampling with uniform prior on theta
#     nsample = 10 # number of samples
#     l_1 = min(theta[j], theta_star[j])  # lower bound
#     l_2 = max(theta[j], theta_star[j])  # upper bound
#     gam = np.copy(ga)
#     theta_tran = np.copy(theta)
#     sample = []
#     for i in xrange(nsample):
#         theta_tran[j] = np.random.uniform(l_1, l_2)    # generate transitional theta
#         for v in xrange(N):
#             gam[v, j] = np.random.binomial(1, ga_prop(v, j, a, gam, theta_tran))
#         sample.append(log_Ising(j, a, np.ones((1, p))[0], gam)*(l_2-l_1))
#     if theta[j] > theta_star[j]:
#         output = np.average(sample)
#     else:
#         output = -np.average(sample)
#     return output

# open file to keep track of log_const
f_const = open(dirname+'/const.txt', 'w')
f_const.close()
# path sampling for normalizing constant with fixed width stopping rule
def log_const_theta(j, theta, theta_star, a, ga):   # log normalizing constant in Hastings ratio for updating theta
    # path sampling with uniform prior on theta
    l_1 = min(theta[j], theta_star[j])  # lower bound
    l_2 = max(theta[j], theta_star[j])  # upper bound
    gam = np.copy(ga)
    theta_tran = np.copy(theta)
    sample = []
    mcsample = np.r_[gam[:, j], theta_tran[j]]
    it = 0  # iterations
    thres = 1000    # low bound for check stopping rule
    while 1:
        it += 1
        theta_tran[j] = np.random.uniform(l_1, l_2)    # generate transitional theta
#         gam[:, j] = np.array([np.random.binomial(1, ga_prop(v, j, a, gam, theta_tran)) for v in xrange(N)])
        for v in xrange(N):
            gam[v, j] = np.random.binomial(1, ga_prop(v, j, a, gam, theta_tran))
        sample.append(log_Ising(j, a, np.ones((1, p))[0], gam)*(l_2-l_1))
        mc = np.r_[gam[:, j], theta_tran[j]]
        mcsample = np.vstack((mcsample, mc))
#         if it > thres:
#             thres += 50
#             e = mcmcse.mcse(mcsample.T)[0]
#             se = mcmcse.mcse(mcsample.T)[1]
#             ssd = np.std(mcsample, 0)
#             with open('const.txt', 'a') as f_const:
#                 pickle.dump(np.average(sample), f_const)
#             if np.prod(se*1.645+1./it < 0.5*ssd): # 90% and epsilon = 0.5
#                 break
#         with open('mcsample.txt', 'w') as f_mcsample:
#             pickle.dump(mcsample, f_mcsample)
        if it > thres:
            thres += 500
            with open(dirname+'/const.txt', 'a') as f_const:
                pickle.dump(np.average(sample), f_const)
        if it > 1000:
            break
    if theta[j] > theta_star[j]:
        output = np.average(sample)
    else:
        output = -np.average(sample)
    return output
    
def ratio_theta(j, theta, theta_star):  # Hastings ratio for updating theta
    theta_max = 2   # theta_max
    if theta_star[j] > theta_max or theta_star[j] < 0:
        output = 0
    elif theta[j] > theta_max or theta[j] < 0:
        output = 1
    else:
        log_output = log_const_theta(j, theta, theta_star, a, ga_cur)+log_Ising(j, 0, theta_star-theta, ga_cur)+log(normpdf(theta[j], theta_cur[j], 1)/normpdf(theta_star[j], theta_cur[j], 1))
        if log_output > 0:
            output = 1
        else:
            output = exp(log_output)
    return output
    
def update_ga(v, j, ga_cur, a):    # metropolis hastings for update gamma
    cur = np.copy(ga_cur)
    temp = np.copy(ga_cur)
    temp[v, j] = np.random.binomial(1, ga_prop(v, j, a, ga_cur, theta_cur))    # proposal walk
    r = ratio_ga(temp[v, j], cur[v, j], S(v, design_m, temp, cov_m[v]), S(v, design_m, cur, cov_m[v]))    # Hastings ratio
    u = np.random.uniform() # generate uniform r.v.
    cur = temp*(r > u)+cur*(r < u)    # update gamma[v, j]
    return cur
    
def update_theta(v, j, theta_cur):   # metropolis hastings for update gamma
    cur = np.copy(theta_cur)
    temp = np.copy(theta_cur)
    temp[j] = np.random.normal(cur[j], 1)  # generate proposal r.v.
    r = ratio_theta(j, cur, temp)    # Hastings ratio
    u = np.random.uniform() # generate uniform r.v.
    cur = temp*(r > u)+cur*(r < u)    # update theta[j]
    return cur

# read in .mat
file = loadmat('data-starplus-04847-v7.mat')


# read data
raw = file['data']  # data
coord = file['meta']['colToCoord'][0, 0].astype('float64')  # col to coordinate index
roi = file['meta']['colToROI'][0, 0]    # anatomical regions
action = file['info']['actionRT'][0] # action time
    
    
# parameters
tr = 3-1  # inference for the third trail    P -> S
press = (action[tr][0][0] > 0)*(action[tr][0][0]/1000.0+8)+(action[tr][0][0] == 0)*(4+8)  # second stimulus on the screen
p = 4   # number of parameters
data = raw[tr, 0]   # abstracted data
N = data.shape[1]  # number of voxels
tp = data.shape[0]   # number of time points
t = np.arange(0, tp/2.0, 0.5) # time sequence
a = log(0.1/(1-0.1))    # external field parameter
G = ['CALC','LIPL','LT','LTRIA','LOPER','LIPS','LDLPFC']    # anatomical interested region
q = 0.8722  # threshold for activation in voxels
rep = 1 # replicates
design_m = design(t, press)

# neighborhood structure
neigh = {}  # dictionary for neighborhood structure
for v in xrange(N):
    neigh.update({v: neig(v)})

# point mass estimates for rho
# sample correlation estimation for rho
# rho_0 = np.zeros((1, N))    # rho(0)
# for v in xrange(N):
#     for t in xrange(tp-1):
#         rho[0, v] += (data[t+1, v] - mean(data[:, v]))*(data[t, v] - mean(data[:, v]))
#     rho[0, v] = rho[0, v]/tp
#     for t in xrange(tp):
#         rho_0[0, v] += (data[t, v] - mean(data[:, v]))**2
#     rho_0[0, v] = rho_0[0, v]/tp
#     rho[0, v] = rho[0, v]/rho_0[0, v]

# Golden region search results for rho
# for v in xrange(N):
#     rho[0, v] = goldensearch(-1, 1, loglike_ar, data[:, v]) # MLE for sigma^2
#     sig[0, v] = sig_y(rho[0, v], data[:, v])  # MLE for sigma^2
    
# Newton methods for rho
rhosig = np.array([Newton(loglike_ar, loglike_ar_der, loglike_ar_hess, [0,1], data[:, v]) for v in xrange(N)])
rho = rhosig[:, 0]
sig = rhosig[:, 1]
# for v in xrange(N):
#     [rho[0, v], sig[0, v]] = Newton(loglike_ar, loglike_ar_der, loglike_ar_hess, [0,1], data[:, v])

# write rho and sig in file
with open(dirname+'/rho.txt', 'w') as f_rho:
    pickle.dump(rho, f_rho)
with open(dirname+'/sig.txt', 'w') as f_sig:
    pickle.dump(sig, f_sig)
    
# covariance matrix given rho estimate by MLE
cov_m = {}
for v in xrange(N):
    cov_m.update({v: cov(t, rho[v])})
    
# write cov in file
with open(dirname+'/cov.txt', 'w') as f_cov:
    pickle.dump(cov_m, f_cov)
    
# posterior analysis
for r in xrange(rep):   # r replicates
    #rho = np.zeros((1, N))   # rho for each voxel
    theta = np.ones((1, p)) # strength of interaction
    ga = {0 : np.zeros((N, p))}    # indicator gamma
    ga[0][:, 0:2] = 1  # first two columns are one's
    for v in xrange(N): # for voxels in anatomical region assign gamma = 1
        if roi[v] in G:
            ga[0][v, 2:4] = 1
    n = 0   # iterations
    thresh = 1000   # lower bound for checking stopping rule
    while n < 1:    # mcmc simulation
        n += 1  # counts
        #rho_cur = rho[n-1, :]   # latest rho
        theta_cur = np.copy(theta[n-1, :])   # latest theta
        ga_cur = np.copy(ga[n-1])  # laste gamma
        
        # update gamma
        for v in xrange(N):
            for j in xrange(p - 2): # first two colums are one's
                ga_cur = update_ga(v, j+2, ga_cur, 0) # update gamma
        ga.update({n: ga_cur})
        # write ga in file
        with open(dirname+'/gamma.txt', 'w') as f_ga:
            pickle.dump(ga, f_ga)

        # update rho
        # avoided when using point mass prior for rho
        #for v in xrange(N):
        #    temp = rho_cur
        #    temp[0, v] = np.random.uniform(-1, 1)  # generate proposal r.v.
        #    r = ratio_rho(temp, rho_cur)    # Hastings ratio
        #    u = np.random.uniform() # generate uniform r.v.
        #    rho_cur = temp*(r > u)+rho_cur*(r < u)    # update rho[v]
#         rho = vstack([rho, rho_cur])    # updates
        
        # update theta
        for j in xrange(p):
            theta_cur = update_theta(v, j, theta_cur)   # update theta
        theta = np.vstack([theta, theta_cur])
        # write theta in file
        with open(dirname+'/theta.txt', 'w') as f_theta:
            pickle.dump(theta, f_theta)

#         if n > thresh:
#             thresh += 1000
             
        

    
# Standard boilerplate to call the main() function to begin
# the program.
# if __name__ == '__main__':
#     main()
    
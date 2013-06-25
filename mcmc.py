#!/usr/bin/python

'''
MCMC updates for parameters with stopping rule

ratio_theta(j, theta, theta_star): hastings ratio for theta
ratio_gamma(v, j, gamma, gamma_star): hastings ratio for gamma
update_theta(j, theta_cur): stepwise update for theta
update_gamma(v, j, gamma_cur): stepwise update for gamma
mcmc_update(theta, gamma): mcmc updates with stopping rule 
'''

# import module
import sys
sys.path.append('/rhome/lgong/mcmcse')
import mcmcse

import pickle
from scipy.stats import norm

import model


#### proposal distribution for gamma ####
# proposal probability
def gamma_prop(v, j, gamma, theta):
    part = np.sum([model.w(coord, v, k)*(1-2*gamma[k, j]) for k in neigh[v]])
    return 1/(1 + exp(-a + theta[j]*part))


#### hastings ratio for theta & gamma ####
# ratio theta
def ratio_theta(j, theta, theta_star):
    log_output = log_const_theta(j, theta, theta_star, ga_cur)+model.log_Ising(j, 0, theta_star-theta, ga_cur)+np.log(norm.pdf(theta[j], theta[j], 0.6)/norm.pdf(theta_star[j], theta[j], 0.6))
    if log_output > 0:
        output = 1
    else:
        output = np.exp(log_output)
    return output
    
# ratio gamma
def ratio_gamma(v, j, gamma, gamma_star): # Hastings ratio for updating gamma
    S_1 = model.S(v, cov_m_inv, gamma_star, data, tp, design_m)
    S_2 = model.S(v, cov_m_inv, gamma, data, tp, design_m)
    return (1+tp)**(-gamma_star[v, j]/2+gamma[v, j]/2)*(S_1/S_2)**(tp/2)
    

#### stepwise update for theta & gamma ####
# update theta
def update_theta(j, theta_cur):
    theta_max = 2   # theta_max
    cur = np.copy(theta_cur)
    temp = np.copy(theta_cur)
    temp[j] = np.random.normal(cur[j], 0.6)  # generate proposal r.v.
    if cur[j] > theta_max or cur[j] < 0:
        cur[j] = temp[j]
    elif temp[j] > theta_max or temp[j] < 0:
        cur[j] = cur[j]
    else:
        r = ratio_theta(j, cur, temp)    # Hastings ratio
        u = np.random.uniform() # generate uniform r.v.
        cur[j] = temp[j]*(r > u)+cur[j]*(r < u)    # update theta[j]
    if cur[j] != temp[j]:
        accept.append(0)
    else:
        accept.append(1)
    with open(dirname+'/accept.txt', 'w') as f_accept:
        pickle.dump(accept, f_accept) 
    return cur

# update gamma
def update_gamma(v, j, gamma_cur):
    cur = np.copy(gamma_cur)
    temp = np.copy(gamma_cur)
    temp[v, j] = np.random.binomial(1, gamma_prop(v, j, gamma_cur, theta_cur))    # proposal walk
    if temp[v, j] != cur[v, j]:
        r = ratio_gamma(v, j, cur, temp)    # Hastings ratio
        u = np.random.uniform() # generate uniform r.v.
        cur = temp*(r > u)+cur*(r < u)    # update gamma[v, j]
    return cur
    

#### mcmc update for theta & gamma ####
def mcmc_update(theta, gamma):
    thresh = 1000   # threshold for checking mcmcse
    n = 0   # start simulation
    comb = np.append(gamma[0].flatten(), theta.flatten())  # storage of all parameters
    
    while 1:
        n += 1  # counts
        theta_cur = np.copy(theta[n-1, :])  # latest theta
        gamma_cur = np.copy(gamma[n-1])  # laste gamma
        
        # update gamma
        for v in range(N):
            for j in range(p - 2): # first two colums are one's
                gamma_cur = update_gamma(v, j+2, gamma_cur) # update gamma
        gamma.update({n: gamma_cur})
        # write gamma in file
        with open(dirname+'/gamma.txt', 'w') as f_gamma:
            pickle.dump(gamma, f_gamma)
        
        # update theta
        for j in range(p):
            theta_cur = update_theta(j, theta_cur)   # update theta
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
            e = mcmcse.mcse(comb.T)[0]
            se = mcmcse.mcse(comb.T)[1]
            ssd = np.std(comb, 0)
            if np.prod(se*1.645+1./n < 0.05*ssd): # 90% and epsilon
                break
                
        
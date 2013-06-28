#!/usr/bin/python

'''
main() for fMRI data analysis with Bayesian spatio-temporal variable selection model

make directories
read in data
set parameters
call modules
'''

# import modules
import sys
import os
import pickle
import numpy as np
from scipy.io import loadmat
from numpy.linalg import inv

import model
import mcmc


# main()
def main():
    # make new dir as input to store results
    if len(sys.argv) >= 2:
        dirname = sys.argv[1]
    os.mkdir(dirname)   # make new directory
    
    np.random.seed(3)   # set random seed
    
    file = loadmat('data-starplus-04847-v7.mat')    # read in .mat
    
    # read data
    raw = file['data']  # data
    coord = file['meta']['colToCoord'][0, 0].astype('float64')  # col to coordinate index
    roi = file['meta']['colToROI'][0, 0]    # anatomical regions
    action = file['info']['actionRT'][0] # action time
    
    # set parameters
    tr = 3-1  # inference for the third trail    P -> S
    press = (action[tr][0][0] > 0)*(action[tr][0][0]/1000.0+8)+(action[tr][0][0] == 0)*(4+8)  # second stimulus on the screen
    p = 4   # number of parameters
    data = raw[tr, 0]   # abstracted data
    N = data.shape[1]  # number of voxels
    tp = data.shape[0]   # number of time points
    a = np.log(0.1/(1-0.1))    # external field parameter
    G = ['CALC','LIPL','LT','LTRIA','LOPER','LIPS','LDLPFC']    # anatomical interested region
    q = 0.8722  # threshold for activation in voxels
    
    # initial values
    theta = np.ones((1, p)) # strength of interaction
    gamma = {0 : np.zeros((N, p))}    # indicator gamma
    gamma[0][:, 0:2] = 1  # first two columns are one's
    for v in range(N): # for voxels in anatomical region assign gamma = 1
        if roi[v] in G:
            gamma[0][v, 2:4] = 1
    
    # estimates
    neigh = model.neighbor(N, coord)    # neighbor structure
    
    rhosig = model.rhosig_mle(data, N)  # MLE for rho and sigma
    rho = rhosig[:, 0]
    sig = rhosig[:, 1]
    with open(dirname+'/rho.txt', 'w') as f_rho:
        pickle.dump(rho, f_rho)
    with open(dirname+'/sig.txt', 'w') as f_sig:
        pickle.dump(sig, f_sig)
        
    [cov_m, cov_m_inv] = model.cov_matrix(rho, N, tp)   # covariance matrix
    with open(dirname+'/cov.txt', 'w') as f_cov:
        pickle.dump(cov_m, f_cov)
            
    design_m = model.design(tp, press)    # design matrix
    
    # update
    mcmc.mcmc_update(theta, gamma, coord, neigh, cov_m_inv, data, tp, design_m, p, N, dirname)
    

if __name__ == '__main__':
    main()
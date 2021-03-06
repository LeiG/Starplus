#!/usr/bin/python

"""
This program uses a spatial Bayesian variable selection model to study
a fMRI experiment. The spBayes model is introduced by Lee et al. (2014)
and the dataset is from the StarPlus experiment (Carpenter et al.
(1999)).

The relative standard deviation fixed-width stopping rule and the
Geweke diagnostics are utilized to terminate the MCMC simulations
separately in order to compare their performances in high dimensional
settings.
"""

import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import posterior
import sys
import os
import pickle
from scipy.io import loadmat

def main():
    """
    The main() reads, pre-processes the data and calls the MCMC updates.
    """

    if len(sys.argv) >= 2:
        dirname = sys.argv[1]
    os.mkdir(dirname)  #make new directory to store results

    np.random.seed(3)  #set random seed

    file = loadmat('data-starplus-04847-v7.mat')  #read in .mat

    # read data
    raw = file['data']
    coord = file['meta']['colToCoord'][0, 0].astype('float64')  #coordinate
    roi = file['meta']['colToROI'][0, 0]    #anatomical regions
    action = file['info']['actionRT'][0]  #action time
    tr = 3-1  #inference for the third trail    P -> S
    totdata = raw[tr, 0]   #total abstracted data
    totN = totdata.shape[1]  #total number of voxels

    # select only the Region Of Interests(ROI)
#    G = ['CALC']
#    G = ['CALC','LIPL','LT','LTRIA','LOPER','LIPS','LDLPFC']
#    voi = [v for v in xrange(totN) if roi[v] in G] #voxels of interests

    # select the entire brain
    voi = [v for v in xrange(totN)] #all voxels


    # set parameters
    press = ((action[tr][0][0] > 0)*(action[tr][0][0]/1000.0+8.0) +
            (action[tr][0][0] == 0)*(4.0+8.0))  #second stimulus on the screen
    p = 4   #number of parameters
    data = raw[tr, 0][:, voi]   #abstracted data
    N = data.shape[1]  #number of voxels
    tp = data.shape[0]   #number of time points
#    a = np.log(0.1/(1.0-0.1))    #external field parameter
#    q = 0.8722  #threshold of activation


    # estimates
    neighbor = posterior.neighbor(N, coord[voi])
    neigh = neighbor[0] #neighborhood structure
    weight = neighbor[1]    #weights
    with open(dirname+'/neigh.txt', 'w') as f_neigh:
        pickle.dump(neigh, f_neigh)
    with open(dirname+'/weight.txt', 'w') as f_weight:
        pickle.dump(weight, f_weight)

    rhosig = posterior.rhosig_mle(data, N)  #MLE for rho and sigma
    rho = rhosig[:, 0]
    sig = rhosig[:, 1]
    with open(dirname+'/rho.txt', 'w') as f_rho:
        pickle.dump(rho, f_rho)
    with open(dirname+'/sig.txt', 'w') as f_sig:
        pickle.dump(sig, f_sig)

    [cov_m, cov_m_inv] = posterior.cov_matrix(rho, N, tp)   #covariance matrix
    with open(dirname+'/cov.txt', 'w') as f_cov:
        pickle.dump(cov_m, f_cov)

    design_m = posterior.design(tp, press)    #design matrix

    # update
#     posterior.mcmc_update(neigh, cov_m_inv, data, np.float(tp), design_m, p, N, dirname)  #use fixedwidth stopping rule
#     posterior.mcmc_ess_update(neigh, cov_m_inv, data, np.float(tp), design_m, p, N, dirname)  #use effective sample size
    posterior.mcmc_diag(neigh, cov_m_inv, data, np.float(tp), design_m, p, N, dirname) # use Geweke diagnostics


if __name__ == '__main__':
    main()

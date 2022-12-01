import logging
import numpy as np
import torch


import os,sys
sys.path.append(os.getcwd())
from helpers.torch_utils import set_seed


class Causal_CCSL(object):
    _logger = logging.getLogger(__name__)

    def __init__(self, num_samples, num_variables, max_lag, device ,prior_mu, prior_sigma, prior_nu, prior_omega):

        self.num_samples = num_samples
        self.num_variables = num_variables
        self.max_lag = max_lag
        self.device = device

        # b_ij^k
        self.prior_mu = prior_mu 
        self.prior_sigma = prior_sigma
        # a_ij,p^k
        self.prior_nu = prior_nu
        self.prior_omega = prior_omega
        
        self.init_prior()

        # result
        self.cluster = [] # Store the cluster result of Xs by the order of input X, which gives us a list with size n
        self.causal_structures = [] # Store the learned causal structure of each group, which should be a list with size q.

        self._logger.debug('Finished building model')

    def init_prior(self):
        # Set up priors

        m = self.num_variables
        mu = self.prior_mu*torch.ones(size=(m,m),device=self.device)
        sigma = self.prior_sigma*torch.ones(size=(m,m),device=self.device)

        # Set the diagonal to zero, since no instantaneous self-cause 
        mu = mu- torch.diag(mu.diag())
        sigma = sigma- torch.diag(sigma.diag()) 
        self.prior_B = [mu,sigma]


        pl = self.max_lag #max_lag
        nu = self.prior_nu*torch.ones(size=(pl,m,m),device=self.device)
        omega = self.prior_omega*torch.ones(size=(pl,m,m),device=self.device)
        self.prior_A = [nu,omega]
          

        q_prime = m
        noise_mu = torch.rand(size=[m,q_prime],device=self.device) # [0,1)
        noise_sigma = torch.rand(size=[m,q_prime],device=self.device) # [0,1)

        noise_prob_ = 1.0/q_prime*torch.ones(size=[m,q_prime-1],device=self.device)
    
        noise_prob_last = torch.ones(size=[m],device=self.device) - torch.sum(noise_prob_,axis=1)
        
        noise_prob = torch.cat((noise_prob_,noise_prob_last.reshape(m,1)),1)

        self.proir_noise = [noise_prob_, noise_mu, noise_sigma]
        self.prior = [self.prior_B, self.prior_A, self.proir_noise]
        
    @property
    def logger(self):
        try:
            return self._logger
        except:
            raise NotImplementedError('self._logger does not exist!')


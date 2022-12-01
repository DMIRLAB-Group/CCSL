import logging
import numpy as np
import copy
import torch
from itertools import chain
from torch import optim
from torch.distributions import Normal,Categorical
from torch.nn.utils import clip_grad_value_
from helpers.analyze_utils import  plot_losses, AUC_score



class Trainer(object):
    """
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, learning_rate, num_iterations, num_output, num_MC_sample, num_total_iterations, num_init_iterations):

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_output = num_output
        self.num_MC_sample = num_MC_sample
        self.num_total_iterations = num_total_iterations
        self.num_init_iterations = num_init_iterations


    def train_model_search(self, model, X,  output_dir, groundtruth_cluster, groundtruth_matrix):
        self.output_dir = output_dir

        # X is shape of (n,Ts,m) # n is the number of total subjects and m is the number of variables.
        n,Ts,m = X.shape

        number_of_group = [] # Record the number of subjects in each group
        # Init the causal strcutures for each subjects, i.e. each subject is a group
        self.init_flag = True
        for it in range(self.num_init_iterations): 
            for i in range(n):
                    self._logger.info("Init Causal Structure for subject #{}".format(i))
                    Xs = X[i]

                    if it>0:
                        G_prior = model.causal_structures[i]
                    else:
                        G_prior = model.prior

                    G_i = self.Causal_Structure_learning(prior_structure=G_prior, Xs=Xs)
                    
                    if it>0:
                        model.causal_structures[i] = G_i
                    else:
                        model.causal_structures.append(G_i)
                        model.cluster.append(i)
                        number_of_group.append(1)
                        
                        plot_losses(self.train_losses,display_mode=False,save_name=output_dir+'/loss_subject_#{}.png'.format(i))

        self.init_flag = False
        last_cluster = copy.deepcopy(model.cluster)
        best_itera = 0
        best_auc = 0
        best_ari = -1
        for itera in range(self.num_total_iterations):

            for i in range(n):
                self._logger.info("Clustering and Causal Structure Learning for subject #{}".format(i))
                Xs = X[i]
                # Clustering
                prob_Cs = self.Causal_CRP(causal_structures=model.causal_structures, number_of_group=number_of_group, Xs=Xs) # prob_Cs is a list with size K, each element is the probability of Cs=1,...,K
            
                self._logger.info("Clustering probabilities:{}".format(prob_Cs))

                # k = np.random.choice(range(len(prob_Cs)),p=prob_Cs) # Note that in python, index starts from 0, so Cs=k= 0,...,K
                k = np.argmax(prob_Cs)

                self._logger.info("Cluster result:{}".format(k))
                # Store the cluster result of Xs
                model.cluster.append(k)
                # the number of subjects in group k should plus 1.
                number_of_group[k] += 1
                # the number of subjects in group p(which is the group Xs belonged before) should subtract 1.
                p = last_cluster[i]
                number_of_group[p] -= 1


                self._logger.info("Updating Causal strcuture of group #{}".format(k))
                
                # Causal structure learning
                G_prior = model.causal_structures[k]
                
                G_k = self.Causal_Structure_learning(prior_structure=G_prior, Xs=Xs)
                self._logger.info("Finished Update")

                # plot_losses(self.train_losses,display_mode=False,save_name=output_dir+'/iteration{}_loss_subject_#{}.png'.format(itera+1,i))

                # Update G_k
                model.causal_structures[k] = G_k

            self._logger.info("Fininshed one big iteration, the {}th iteration cluster result: {}".format(itera+1,model.cluster[-n:]))
            last_cluster = model.cluster[-n:]
            model.cluster.append('|') # Separater

            # Calculate performance
            averaged_AUC = 0

            cluster = model.cluster[-1-(n):-1]
            for c in set(cluster):
                    parameters = model.causal_structures[c]

                    estimate_B = np.abs(parameters[0][0].numpy())
                    estimate_A = np.abs(parameters[1][0].numpy())

                    # Normalize to [0,1]
                    estimate_B = estimate_B / np.max(estimate_B) 
                    estimate_A = estimate_A / np.max(estimate_A)

                    estimate_graph = np.concatenate([estimate_B.reshape(1,m,m), estimate_A])

                    # may be not right in all case.
                    true_cluster_index = groundtruth_cluster[ cluster.index(c) ]

                    group_dag =  groundtruth_matrix[true_cluster_index]
                    Score = AUC_score(estimate_graph,group_dag)

                    averaged_AUC += Score['AUC']

            averaged_AUC = averaged_AUC/ len(set(cluster))
            self._logger.info('Averaged AUC: {}'.format(averaged_AUC))

            from sklearn.metrics import adjusted_rand_score
            ari=adjusted_rand_score(groundtruth_cluster, cluster) 
            self._logger.info('Adjusted Rand index，ARI: {}'.format(ari)) 

            if ari>0.5 and averaged_AUC > best_auc:
                best_itera = itera
                best_ari = ari
                best_auc = averaged_AUC


        self._logger.info('Best itear: {}'.format(best_itera+1))         
        self._logger.info('Best AUC: {}'.format(best_auc)) 
        self._logger.info('Best ARI: {}'.format(best_ari)) 

    def train_model(self, model, X,  output_dir):
        self.output_dir = output_dir

        # X is shape of (n,Ts,m) # n is the number of total subjects and m is the number of variables.
        n,Ts,m = X.shape

        number_of_group = [] # Record the number of subjects in each group
        # Init the causal strcutures for each subjects, i.e. each subject is a group
        self.init_flag = True
        for it in range(self.num_init_iterations): 
            for i in range(n):
                    self._logger.info("Init Causal Structure for subject #{}".format(i))
                    Xs = X[i]

                    if it>0:
                        G_prior = model.causal_structures[i]
                    else:
                        G_prior = model.prior

                    G_i = self.Causal_Structure_learning(prior_structure=G_prior, Xs=Xs)
                    
                    if it>0:
                        model.causal_structures[i] = G_i
                    else:
                        model.causal_structures.append(G_i)
                        model.cluster.append(i)
                        number_of_group.append(1)
                        
                        plot_losses(self.train_losses,display_mode=False,save_name=output_dir+'/loss_subject_#{}.png'.format(i))

        self.init_flag = False
        last_cluster = copy.deepcopy(model.cluster)
        crp_total_loss = []
        for itera in range(self.num_total_iterations):
            crp_loss = 0
            for i in range(n):
                self._logger.info("Clustering and Causal Structure Learning for subject #{}".format(i))
                Xs = X[i]
                # Clustering
                prob_Cs = self.Causal_CRP(causal_structures=model.causal_structures, number_of_group=number_of_group, Xs=Xs) 
            
                self._logger.info("Clustering probabilities:{}".format(prob_Cs))

                
                k = np.argmax(prob_Cs)

                self._logger.info("Cluster result:{}".format(k))
                model.cluster.append(k)
                # the number of subjects in group k should plus 1.
                number_of_group[k] += 1
                # the number of subjects in group p(which is the group Xs belonged before) should subtract 1.
                p = last_cluster[i]
                number_of_group[p] -= 1


                self._logger.info("Updating Causal strcuture of group #{}".format(k))
                
                # Causal structure learning
                G_prior = model.causal_structures[k]
                
                G_k = self.Causal_Structure_learning(prior_structure=G_prior, Xs=Xs)
                self._logger.info("Finished Update")

                # plot_losses(self.train_losses,display_mode=False,save_name=output_dir+'/iteration{}_loss_subject_#{}.png'.format(itera+1,i))

                # Update G_k
                model.causal_structures[k] = G_k
                crp_loss += self.train_losses[0]
            crp_total_loss.append(crp_loss)
            self._logger.info("Fininshed one big iteration, the {}th iteration cluster result: {}".format(itera+1,model.cluster[-n:]))
            last_cluster = model.cluster[-n:]
            model.cluster.append('|') # Separater
        plot_losses(crp_total_loss,display_mode=False,save_name=output_dir+'/CRP_total_loss.png')
            
            

    def Causal_CRP(self,causal_structures,number_of_group,Xs):

        Probs = []
        K = len(causal_structures)
        
        # For avoiding Numeric Overflow
        constant_C = torch.tensor([float('inf')],device=causal_structures[0][0][0].device)
        cnt_group = K
        for k in range(K):
            if number_of_group[k]<1:
                cnt_group -= 1
                continue
            
            distribution_B = Normal(loc=causal_structures[k][0][0], scale=causal_structures[k][0][1])
            distribution_A = Normal(loc=causal_structures[k][1][0], scale=causal_structures[k][1][1])

            m = causal_structures[k][0][0].shape[0] # num_variables
            noise_prob_ = causal_structures[k][2][0]
            noise_prob_last = torch.ones(size=[m],device=noise_prob_.device) - torch.sum(noise_prob_,axis=1)
            noise_prob = torch.cat((noise_prob_,noise_prob_last.reshape(m,1)),1)

            mix  = Categorical(probs=noise_prob)
            comp = Normal(loc=causal_structures[k][2][1] , scale=causal_structures[k][2][2])

            distribution = [distribution_B,distribution_A, mix, comp]
            
            constant_C = torch.min(torch.abs( self.log_likelihood_fun(distribution=distribution,X=Xs) ),constant_C)
            

        for k in range(K):
            if number_of_group[k]<1:
                Probs.append(0.0)
                continue

            Pi_k = torch.tensor(1.0/cnt_group,device=causal_structures[0][0][0].device) # P(c_s=k)
            M = self.num_MC_sample
            
            distribution_B = Normal(loc=causal_structures[k][0][0], scale=causal_structures[k][0][1])
            distribution_A = Normal(loc=causal_structures[k][1][0], scale=causal_structures[k][1][1])

            m = causal_structures[k][0][0].shape[0] # num_variables
            noise_prob_ = causal_structures[k][2][0]
            noise_prob_last = torch.ones(size=[m],device=noise_prob_.device) - torch.sum(noise_prob_,axis=1)
            noise_prob = torch.cat((noise_prob_,noise_prob_last.reshape(m,1)),1)

            mix  = Categorical(probs=noise_prob)
            comp = Normal(loc=causal_structures[k][2][1] , scale=causal_structures[k][2][2])

            distribution = [distribution_B,distribution_A, mix, comp]
            
            log_p_Xs = 0.0
            for t in range(M):
                log_p_Xs +=  self.log_likelihood_fun(distribution=distribution,X=Xs) 
            log_p_Xs = log_p_Xs/M

            p_Xs = torch.exp(log_p_Xs / constant_C)
           
            P_k = p_Xs*Pi_k

            Probs.append(P_k.item())

        return np.array(Probs)/np.array(Probs).sum()



    def Causal_Structure_learning(self,prior_structure,Xs):
        
        train_losses = []

        
        # Init prior
        
        prior_distribution_B = Normal(loc=prior_structure[0][0], scale=prior_structure[0][1])
        prior_distribution_A = Normal(loc=prior_structure[1][0], scale=prior_structure[1][1])
        
        prior_distribution = [prior_distribution_B,prior_distribution_A]
        
        # Init posterior
        posterior_structure = copy.deepcopy(prior_structure)
        
        for para in sum(posterior_structure,[]):
            para.requires_grad = True
        
        posterior_distribution_B = Normal(loc=posterior_structure[0][0], scale=posterior_structure[0][1])
        posterior_distribution_A = Normal(loc=posterior_structure[1][0], scale=posterior_structure[1][1])
        
        m = prior_structure[0][0].shape[0] # num_variables
        noise_prob_ = posterior_structure[2][0]
        noise_prob_last = torch.ones(size=[m],device=noise_prob_.device) - torch.sum(noise_prob_,axis=1)
        noise_prob = torch.cat((noise_prob_,noise_prob_last.reshape(m,1)),1)

        mix  = Categorical(probs=noise_prob)
        comp = Normal(loc=posterior_structure[2][1] , scale=posterior_structure[2][2])

        posterior_distribution = [posterior_distribution_B,posterior_distribution_A, mix, comp]
        

            
        optimizer = optim.Adam(params= sum(posterior_structure,[]), lr=self.learning_rate)
        for iteration in range(self.num_iterations):

            optimizer.zero_grad()

            loss = self.loss_fun(prior_distribution,posterior_distribution,Xs)

            if loss < 0.0: # if loss is negative, early stopping
                self._logger.info("The log_likelihood is negative, early stopping.")
                break
            
            if torch.isnan(loss): # if loss is NAN, break
                self._logger.info("!!!! Loss is NAN, check the generated data and configuration.")
                break
            loss.backward(retain_graph=True)

            # Clipping Gradient for parameters
            clip_grad_value_(sum(posterior_structure,[]),clip_value=5.0)
            with torch.no_grad():
                # In case NAN
                for para in sum(posterior_structure,[]):
                    para.grad[torch.isnan(para.grad)] = 0.0
            


            train_losses.append(loss.item())

            if(iteration% self.num_output==0):
                self._logger.info("Iteration {} , loss:{}".format(iteration,loss.item()))

            optimizer.step()
            with torch.no_grad():
                
                #Pi should >0 and <1
                posterior_structure[2][0].data = torch.clamp(posterior_structure[2][0],min=0.0,max=1.0)

                # scale should >0, and  loss will become NAN when scale is 0 so we simply set the minimal scale is 1e-5.
                posterior_structure[0][1].data = torch.clamp(posterior_structure[0][1],min=1e-5)
                posterior_structure[1][1].data = torch.clamp(posterior_structure[1][1],min=1e-5)
                posterior_structure[2][2].data = torch.clamp(posterior_structure[2][2],min=1e-5)

                
                # Set the data of the diagonal in B with 0
                for i in range(posterior_structure[0][1].shape[0]):
                    posterior_structure[0][0][i,i] = 0.0 #loc
                    posterior_structure[0][1][i,i] = 0.0 #scale

        self.train_losses = train_losses
        for para in sum(posterior_structure,[]):
            para.requires_grad = False

        return posterior_structure

    
    def loss_fun(self,prior,posterior,X):
        """
        return: the negative ELBO
        """

        # Calculating L_kl
        from torch.distributions.kl import kl_divergence
        KL_B = kl_divergence(p=posterior[0], q=prior[0])
        # Replace the diagonal elements(NAN) with zero
        KL_B[torch.isnan(KL_B)] = 0
        
        KL_A = kl_divergence(p=posterior[1], q=prior[1])
        KL_A[torch.isnan(KL_A)] = 0
        L_kl = KL_B + KL_A


        
        L_ell = self.log_likelihood_fun(distribution=posterior,X=X)
        if L_ell >0.0 :#(torch.tensor(L_ell)>0.0).sum() > 0.0:
            return torch.tensor([float('-inf')])


        ELBO =  -L_kl.sum() + L_ell
        loss = -ELBO

        if self.init_flag:
            return loss 
        else:
            return loss #+ 2*L1_loss + 1e4*_notear_constraint(B)


    def log_likelihood_fun(self,distribution,X):
        
        from torch.distributions import MixtureSameFamily
        gmm = MixtureSameFamily(distribution[2],distribution[3])
        # See E.q. (15)
        log_P_x1 = ( gmm.log_prob(X[0])).sum()


        B = distribution[0].rsample()
        A = distribution[1].rsample()
        
        Ts = X.shape[0]
        m = X.shape[1]
        pl = A.shape[0]

        log_P_xp = []
        for t in range(2,pl+1):

            tmp_noise = torch.matmul((torch.eye(m,device=B.device)-B),X[t-1])
            for p in range(1,t): 
                tmp_noise -=  torch.matmul(A[p-1], X[t-1-p])
            
            log_P_xp.append( t*torch.log( torch.abs(torch.det( torch.eye(m,device=B.device)-B ))) +  (gmm.log_prob(tmp_noise)).sum() )

        self.log_P_xp = log_P_xp

        tmp_noise = torch.matmul( (torch.eye(m,device=B.device)-B),X[pl:Ts+1].T)
        for p in range(1,pl+1):
            tmp_noise -=  torch.matmul(A[p-1],X[pl-p:Ts-p].T)
        
        log_P_xT = ((pl+1)*torch.log(torch.abs(torch.det( torch.eye(m,device=B.device)-B ))) + (gmm.log_prob(tmp_noise.T)).sum(dim=1) )


        self.log_P_xT = log_P_xT.sum()
        return (self.log_P_xT + sum(self.log_P_xp) + log_P_x1)/Ts


            
    def log_and_save_intermediate_outputs(self,model):
        # may want to save the intermediate results
        
        cluster = []
        for tmp in model.cluster[:-1][::-1]:
            if tmp == '|':
                break
            cluster.append(tmp)

        for c in set(cluster):
            parameters = model.causal_structures[c]
            np.save(self.output_dir+'/estimated_parameters_group_{}.npy'.format(c+1), np.array(parameters,dtype=object) ) 
            




















# the first version: init the group according the order of input, so the result will heavily depend on the hyper-parameter alpha.
        
    # def train_model(self, model, X,  output_dir):
    #     self.output_dir = output_dir
        
    #     # X is shape of (n,Ts,m) # n is the number of total subjects and m is the number of variables.
    #     n,Ts,m = X.shape

    #     for itera in range(self.num_total_iterations):

            
    #         for i in range(n):
    #             self._logger.info("Clustering and Causal Structure Learning for subject #{}".format(i+1))
    #             Xs = X[i]
    #             # Clustering
    #             prob_Cs = self.Causal_CRP(causal_structures=model.causal_structures, alpha=model.alpha, Xs=Xs) # prob_Cs is a list with size K+1, each element is the probability of Cs=1,...,K+1
    #             self._logger.info("Clustering probabilities:{}".format(prob_Cs))

    #             # k = np.random.choice(range(len(prob_Cs)),p=prob_Cs) # Note that in python, index starts from 0, so Cs=k= 0,...,K
    #             k = np.argmax(prob_Cs)

    #             self._logger.info("Cluster result:{}".format(k+1))
    #             # Store the cluster result of Xs
    #             model.cluster.append(k)

    #             self._logger.info("Updating Causal strcuture of group #{}".format(k+1))
    #             # Causal structure learning
    #             if k<len(model.causal_structures):
    #                 G_prior = model.causal_structures[k]
    #             else:
    #                 # Means a new cluster
    #                 G_prior = model.prior

    #             G_k = self.Causal_Structure_learning(prior_structure=G_prior, Xs=Xs)
    #             self._logger.info("Finished Update")

    #             plot_losses(self.train_losses,display_mode=False,save_name=output_dir+'/iteration{}_loss_subject_#{}.png'.format(itera+1,i+1))

    #             # Update G_k
    #             if k<len(model.causal_structures):
    #                 model.causal_structures[k] = G_k
    #             else:
    #                 model.causal_structures.append(G_k)

    #         self._logger.info("Fininshed one big iteration, the {}th iteration cluster result: {}".format(itera+1,model.cluster))
    #         model.cluster.append('|') # Separater
            


    # def Causal_CRP(self,causal_structures,alpha,Xs):

    #     Probs = []
    #     K = len(causal_structures)
    #     constant_C = 100 # the estimate log_p_Xs will be negative and may < -200, we want the log_p_Xs in range like [-10,0] so that the torch.exp(log_p_Xs) will in range [4.5400e-05,1]
    #     for k in range(K):
    #         Pi_k = torch.tensor(1.0/K,device=causal_structures[0][0].loc.device) # P(c_s=k)
    #         M = self.num_MC_sample
    #         G_k = causal_structures[k]

    #         p_Xs = 0
    #         for t in range(M):
    #             log_p_Xs = self.log_likelihood_fun(distribution=G_k,X=Xs)
    #             p_Xs += torch.exp(log_p_Xs / constant_C)
    #         p_Xs = p_Xs/M

    #         P_k = p_Xs*Pi_k

    #         Probs.append(P_k.item())

    #     Probs.append(alpha)
    #     return np.array(Probs)/np.array(Probs).sum()

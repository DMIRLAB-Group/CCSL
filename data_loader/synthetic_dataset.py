import logging
import numpy as np
import os,sys
sys.path.append(os.getcwd())
from helpers.torch_utils import set_seed
from helpers.analyze_utils import plot_timeseries


class SyntheticDataset(object):
    """
    A Class for generating data.
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, num_groups, num_subjects_per_group,  num_samples, num_variables, max_lag, noise_distribution='MoG', num_gaussian_component=5):
        """
        Args:
            num_groups: number of groups, i.e. q
            num_subjects_per_group: number of subjects in each group, and note that num_subjects_per_group*num_groups = n
            num_samples: number of samples for each subject, i.e. Ts
            num_variables: number of observed variables, i.e. m 
            max_lag: the maximal lag, i.e. pl
            
            noise_distribution: the distribution of noise in each group, default Mixture of Gaussian
            num_gaussian_component: number of Gaussian component in the Mixture of Gaussian, i.e. q'
            
        """
        
        self.num_groups = num_groups
        self.num_subjects_per_group = num_subjects_per_group
        self.num_samples = num_samples
        self.num_variables = num_variables
        self.max_lag = max_lag

        self.noise_distribution = noise_distribution
        self.num_gaussian_component = num_gaussian_component

        self._setup()
        self._logger.debug('Finished setting up dataset class')

    def _setup(self):
        self.matrix = SyntheticDataset.simulate_all_dag(self.num_groups, self.num_variables, self.max_lag)

        self.X = SyntheticDataset.simulate_all_sem(self.matrix, self.num_groups, self.num_subjects_per_group, self.num_samples, self.num_variables, self.noise_distribution, self.num_gaussian_component)

    
    
    @staticmethod
    def simulate_all_dag(num_groups, num_variables, max_lag):

        """ Simulate all DAG.
            In each group we consider the following generation:
                For each lag(0 means only instantaneous effect), we generate a DAG with m nodes, then we have (pl+1) graphs.

            Returns:
                matrix: a list with size num_groups, each element is a binary tensor with shape (pl+1,m,m)

        """
        matrix = []

        for group in range(num_groups):
            tmp_graphs = []
            for lag in range(max_lag+1):
                B = np.zeros(shape=(num_variables,num_variables))

                while abs(B).sum()<3: # in case B is all zeros
                    B = SyntheticDataset.simulate_dag(num_variables)

                tmp_graphs.append(B)
            matrix.append( np.array(tmp_graphs))

        return matrix

    
    @staticmethod
    def simulate_dag( num_variables, graph_type='ER', prob=0.3):

        """Simulate random DAG with some expected number of edges.
        Args:
                num_variables (int): num of nodes
                prob (float): the probability of edges
                graph_type (str): ER, SF, BP
        Returns:
                B (np.ndarray): [d, d] binary adj matrix of DAG
        """

        import igraph as ig
        def _random_permutation(M):
            # np.random.permutation permutes first axis only
            P = np.random.permutation(np.eye(M.shape[0]))
            return P.T @ M @ P

        def _random_acyclic_orientation(B_und):
            return np.tril(_random_permutation(B_und), k=-1)

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        if graph_type == 'ER':
            # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=num_variables, p=prob)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)

        else:
            raise ValueError('unknown graph type')
        B_perm = _random_permutation(B)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        return B_perm


    @staticmethod
    def simulate_all_sem(matrix, num_groups, num_subjects_per_group, num_samples, num_variables, noise_distribution='MoG', num_gaussian_component=5):
        """ Simulate data for all subjects.

            Inputs:
                matrix: a list with size num_groups, each element is a binary tensor with shape (pl+1,m,m)
            
            In each group k, we consider the following generation:
                With the graph k(i.e. matrix[k]),  we generate data with shape (Ts,m) for each subject, then we have a numpy.array with shape(num_subjects_per_group,Ts,m)
                Note that here DAG[i,j]=1 denotes j->i.

            Returns:
                X: a list with size num_groups, each element is a numpy.array with shape(num_subjects_per_group,Ts,m)

        """
        X = []
        for k in range(num_groups):

            graph = matrix[k]
            tmp_data = []

            
            mu_k_low, mu_k_high = 0.1, 0.4
            nu_k_low, nu_k_high = 0.1, 0.4
            
            m = num_variables
            sigma_k = np.random.uniform(low=0.01,high=0.1,size=(m,m))
            mu_k = np.random.uniform(low=mu_k_low,high=mu_k_high,size=(m,m))
            B_k = np.random.normal(loc=mu_k,scale=sigma_k)*graph[0]

            
            max_lag = graph.shape[0]-1
            if max_lag>0:
                omega_k = np.random.uniform(low=0.01,high=0.1,size=(max_lag,m,m))
                nu_k = np.random.uniform(low=nu_k_low,high=nu_k_high,size=(max_lag,m,m))
                A_k = np.random.normal(loc=nu_k,scale=sigma_k)*graph[1:]
            else:
                A_k = None
            

            
            q_prime = num_gaussian_component # q'
                
            Pi_k_prime = np.random.uniform(low=0.3,high=0.6,size=q_prime)
            Pi_k_prime = Pi_k_prime/Pi_k_prime.sum()
            
            Mu_k_prime = np.random.uniform(low=0.4,high=0.6,size=q_prime) - np.random.randint(low=0,high=2,size=q_prime) # Uniform(-0.6,-0.4) U  Uniform(0.4,0.6)

            Sigma_k_prime = np.random.uniform(low=0.2,high=0.5,size=q_prime) 

            for s in range(30): 
                
                data = SyntheticDataset.simulate_sem(max_lag, num_samples, num_variables, B_k, A_k, noise_distribution, Pi_k_prime, Mu_k_prime, Sigma_k_prime)
                tmp_data.append(data)
            tmp_data = tmp_data[:num_subjects_per_group]
            X.append(np.array(tmp_data))

        return X


    @staticmethod         
    def simulate_sem(max_lag, num_samples, num_variables, B_k, A_k, noise_distribution, Pi_k_prime, Mu_k_prime, Sigma_k_prime):


        burn_in = 200
        T = burn_in
        m = num_variables


        if noise_distribution == 'MoG':

            noise = np.zeros(shape=(T,m))

            for i in range(m):
                for t in range(T):     
                    # Mixture of Gaussian
                    k_prime = np.random.choice(range(len(Pi_k_prime)),p=Pi_k_prime)
                    noise[t,i] = np.random.normal(loc=Mu_k_prime[k_prime],scale=Sigma_k_prime[k_prime],size=1)
       
        else:
            raise ValueError('Undefined noise_distribution type')

        data = np.zeros(shape=(T,m))


        data[0] = np.matmul(np.linalg.inv((np.eye(m)-B_k)), noise[0])


        for t in range(2,max_lag+1):
                
            tmp_t = np.zeros(data[0].shape) 
            for lag in range(1,t):
                
                A_k_p = A_k[lag-1]
                tmp_t += np.matmul( A_k_p , data[t-1 -lag])

            tmp_t += noise[t-1]

            data[t-1] = np.matmul(np.linalg.inv((np.eye(m)-B_k)), tmp_t)


        for t in range(max_lag+1,T+1):
            
            tmp_t = np.zeros(data[0].shape) 
            for lag in range(1,max_lag+1):
                
                A_k_p = A_k[lag-1]
                tmp_t += np.matmul( A_k_p , data[t-1 -lag])

            tmp_t += noise[t-1]

            data[t-1] = np.matmul(np.linalg.inv((np.eye(m)-B_k)), tmp_t)
    

        return  data[-num_samples:]

    def save_dataset(self, output_dir):
        

        for q in range(self.num_groups):

            np.save(output_dir+'/group{}_groudtruth_matrix.npy'.format(q+1),self.matrix[q])
            np.save(output_dir+'/group{}_data.npy'.format(q+1),self.X[q])



if __name__ == '__main__':

    set_seed(2020)
    
    num_groups, num_subjects_per_group,  num_samples, num_variables, max_lag = 2, 10, 20, 5, 2

    dataset = SyntheticDataset(num_groups, num_subjects_per_group,  num_samples, num_variables, max_lag)

    print(dataset.num_groups)
    print(dataset.matrix[0].shape)
    print(dataset.X[0].shape) # the first group,(n_subjects,T,n_variables)



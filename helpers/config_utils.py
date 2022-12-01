import sys
import yaml
import argparse
from helpers.torch_utils import get_device


def load_yaml_config(path, skip_lines=0):
    with open(path, 'r') as infile:
        for i in range(skip_lines):
            # Skip some lines (e.g., namespace at the first line)
            _ = infile.readline()

        return yaml.safe_load(infile)


def save_yaml_config(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def get_args():
    parser = argparse.ArgumentParser()

    ##### General settings #####
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed')

    parser.add_argument('--device',
                        default=get_device(),
                        help='Running device')

    ##### Dataset settings #####
    parser.add_argument('--num_groups',
                        type=int,
                        default=2,
                        help='Number of groups')

    parser.add_argument('--num_subjects_per_group',
                        type=int,
                        default=5,
                        help='Number of subjects in each group')

    parser.add_argument('--num_samples',
                        type=int,
                        default=60,
                        help='Number of sample size for each subject') 
                        
    parser.add_argument('--num_variables',
                        type=int,
                        default=5,
                        help='Number of observed variables')
    
    parser.add_argument('--max_lag',
                        type=int,
                        default=1,
                        help='Number of maximal time lag')


    ##### Model settings #####
    parser.add_argument('--prior_mu',
                        type=float,
                        default=0.0,
                        help='the mean parameter of Normal distribution, which is the prior over b_ij^k')

    parser.add_argument('--prior_sigma',
                        type=float,
                        default=0.1,
                        help='the standard deviation parameter of Normal distribution, which is the prior over b_ij^k')

    parser.add_argument('--prior_nu',
                        type=float,
                        default=0.0,
                        help='the mean parameter of Normal distribution, which is the prior over a_ij,p^k')
    
    parser.add_argument('--prior_omega',
                        type=float,
                        default=0.1,
                        help='the standard deviation parameter of Normal distribution, which is the prior over a_ij,p^k')
                
    ##### Training settings #####
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate for optimizer') 


    parser.add_argument('--num_iterations',
                        type=int,
                        default=500,
                        help='Number of iterations for Variation Inference')

    parser.add_argument('--num_output',
                        type=int,
                        default=10,
                        help='Number of iterations to display information')

                        
    parser.add_argument('--num_MC_sample',
                        type=int,
                        default=30,
                        help='Number of samples during the Monte Carlo integration') 


    parser.add_argument('--num_total_iterations',
                        type=int,
                        default=30,
                        help='Number of the outside iterations')

    parser.add_argument('--num_init_iterations',
                        type=int,
                        default=1,
                        help='Number of the iterations for initializing each group')


    ##### Other settings #####

    return parser.parse_args(args=sys.argv[1:])

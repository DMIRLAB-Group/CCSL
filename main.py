#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pytz import timezone
from datetime import datetime
import numpy as np
import torch


from data_loader.synthetic_dataset import SyntheticDataset
from data_loader.real_dataset import RealDataset
from models.Causal_CCSL import Causal_CCSL
from trainers.trainer import Trainer


from helpers.config_utils import save_yaml_config, get_args
from helpers.log_helper import LogHelper
from helpers.torch_utils import set_seed
from helpers.dir_utils import create_dir
from helpers.analyze_utils import  plot_timeseries, plot_recovered_graph, plot_ROC_curve, AUC_score, F1


def synthetic():
    
    np.set_printoptions(precision=3)
    

    args = get_args()


    output_dir = 'output/{}'.format(datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                    level_str='INFO')
    _logger = logging.getLogger(__name__)
    

    save_yaml_config(args, path='{}/config.yaml'.format(output_dir))


    # Reproducibility
    set_seed(args.seed)

    # Get dataset
    _logger.info('Generating dataset...')
    dataset = SyntheticDataset(args.num_groups, args.num_subjects_per_group,  args.num_samples, args.num_variables, args.max_lag)
    # Save dataset
    dataset.save_dataset( output_dir=output_dir)
    _logger.info('Finished generating dataset')

    # Look at data
    _logger.info('The shape of data (num_groups ,num_subjects, Ts, num_variables): {}, {}'.format(len(dataset.X),dataset.X[0].shape))
    for q in range(args.num_groups):
        plot_timeseries(dataset.X[q],'Group {}'.format(q),display_mode=False,save_name=output_dir+'/group{}_timeseries_X.png'.format(q+1))

    # Shuffle the order of subjects
    data = np.concatenate(dataset.X)
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    shuffled_cluster = np.concatenate([ [i]*args.num_subjects_per_group for i in range(args.num_groups) ]) [permutation]
    _logger.info('The groudtruth clusters: {}'.format(shuffled_cluster))

    # Init model
    model = Causal_CCSL(args.num_samples, args.num_variables, args.max_lag, args.device, args.prior_mu, args.prior_sigma, args.prior_nu, args.prior_omega)

    trainer = Trainer(args.learning_rate, args.num_iterations, args.num_output, args.num_MC_sample, args.num_total_iterations, args.num_init_iterations)

    input_X = torch.tensor(shuffled_data,dtype=torch.float32,device=args.device)

    # trainer.train_model(model=model, X = input_X, output_dir=output_dir)
    trainer.train_model_search(model=model, X = input_X, output_dir=output_dir, groundtruth_cluster=shuffled_cluster,groundtruth_matrix=dataset.matrix)

    # Save result
    trainer.log_and_save_intermediate_outputs(model)

    _logger.info('Finished training model')
    _logger.info('The groudtruth clusters: {}'.format(shuffled_cluster))
    # Calculate performance
    averaged_AUC = 0

    cluster = model.cluster[-1-(args.num_groups* args.num_subjects_per_group):-1]
    for c in set(cluster):
            parameters = model.causal_structures[c]

            estimate_B = np.abs(parameters[0][0].numpy())
            estimate_A = np.abs(parameters[1][0].numpy())

            # Normalize to [0,1]
            estimate_B = estimate_B / np.max(estimate_B) 
            estimate_A = estimate_A / np.max(estimate_A)

            estimate_graph = np.concatenate([estimate_B.reshape(1,args.num_variables,args.num_variables), estimate_A])

            _logger.info('Calculating Performance on the estimated causal structure of Group {}'.format(c))


            true_cluster_index = shuffled_cluster[ cluster.index(c) ]

            group_dag =  dataset.matrix[true_cluster_index]
            Score = AUC_score(estimate_graph,group_dag)
            _logger.info('Performance on groundtruth group {}'.format(true_cluster_index))
            _logger.info('\n        fpr:{} \n        tpr:{}\n thresholds:{}\n AUC:{}'.format(Score['fpr'],Score['tpr'],Score['thresholds'],Score['AUC']))
            plot_ROC_curve(estimate_graph,group_dag,display_mode=False,save_name=output_dir+'/ROC_Curve_group{}_vs_groundtruth{}.png'.format(c,true_cluster_index))
            plot_recovered_graph(estimate_graph,group_dag,title='group{}_vs_groundtruth{}'.format(c+1,true_cluster_index),display_mode=False,save_name=output_dir+'/estimated_group{}_vs_groundtruth{}.png'.format(c,true_cluster_index))

            averaged_AUC += Score['AUC']

    averaged_AUC = averaged_AUC/ len(set(cluster))
    _logger.info('Averaged AUC: {}'.format(averaged_AUC))

    from sklearn.metrics import adjusted_rand_score
    ari=adjusted_rand_score(shuffled_cluster, cluster) 
    _logger.info('Adjusted Rand index，ARI: {}'.format(ari)) 

    _logger.info('All Finished!')



if __name__ == '__main__':

    synthetic()


from glob import glob
import argparse
import os
import copy
import json

import pandas as pd
import numpy as np
from GPyOpt.methods import BayesianOptimization
import torch

from run_experiment import main as run_experiment

bounds = [
    {'name': "learning_rate",     'type': 'continuous', 'domain': (0.0000001,0.1)},
    {'name': 'weight_decay',      'type': 'continuous', 'domain': (0.00001,1)},
    {'name': 'batch_size',        'type': 'discrete',   'domain': [32,64,128,256,512,1024]},
    {'name': "n_convolutions",    'type': 'discrete',   'domain': range(1,15)},
    {'name': 'convolutions_dim',  'type': 'discrete',   'domain': [4,8,16,32,64,128,256,512,1024]},
    {'name': "n_hidden_layers",   'type': 'discrete',   'domain': range(1,10)},
    {'name': 'hidden_layers_dim', 'type': 'discrete',   'domain': [32,64,128,256,512,1024]}
]

def main(args):
    def search(x):
        largs = copy.deepcopy(args)
        largs.learning_rate      = float(x[:, 0])
        largs.weight_decay       = float(x[:, 1])
        largs.batch_size         = int(x[:, 2])
        largs.n_convolutions     = int(x[:, 3])
        largs.convolutions_dim   = int(x[:, 4])
        largs.n_hidden_layers    = int(x[:, 5])
        largs.hidden_layers_dim  = int(x[:, 6])
        largs.model = 'gcn'
        
        print(x)

        save_path = '{save_dir}/'.format(save_dir=largs.output_directory)

        model_iter = 1
        while os.path.exists(os.path.join(save_path, f'model_iter_%03d' % model_iter)):
            model_iter += 1
        largs.output_directory = os.path.join(save_path, f'model_iter_%03d' % model_iter)
        
        run_experiment(largs)

        metrics = pd.concat([pd.read_csv(metrics) for metrics in glob(largs.output_directory+'/metrics*')])
        return metrics['RMSE'].mean()

    np.random.seed(args.random_seed)
    
    # Ensure path is writable
    save_path = '{save_dir}/'.format(save_dir=args.output_directory)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # Save run options
    json.dump(vars(args), open(os.path.join(save_path, f'config.json'), 'w'))

    optimizer = BayesianOptimization(f=search, domain=bounds,
        model_type='GP',
        acquisition_type ='EI',
        acquisition_jitter = 0.05,
        initial_design_numdata = 25,
        exact_feval = True,
        maximize = False,
        verbosity = True)
    
    optimizer.run_optimization(max_iter=100)
    optimizer.save_report(os.path.join(save_path, f'optimization_report.txt'))
    optimizer.save_evaluations(os.path.join(save_path, f'optimization_evals.csv'))
    optimizer.save_models(os.path.join(save_path, f'optimization_models.csv'))

# python ./run_optimization.py -t tg -rp ./data -o ./results/tuning -v True 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Tuning of model.')
    parser.add_argument('-t', '--target', type=str, required=True, help='Specify which endpoint dataset to run the model on. Options are tg, tm, or density.')
    parser.add_argument('-rp', '--raw_path', type=str, required=True, help='Specify path to raw data directory')
    parser.add_argument('-v', '--verbose', type=bool, help='If True, gives detailed status of model while running (eg. fold number, loss per epoch, time taken, etc). Default is False', default=False)
    parser.add_argument('-rs', '--random_seed', type=int, help='Initialize the random seed', default=0)
    parser.add_argument('-o', '--output_directory', type=str, required=True, help='Specify directory of folder to save at, containing models saved at ./fold{fold_number}.pt, results saved at ./results{fold_number}.csv, convenient performance metrics saved at ./metrics{fold_number}.csv, and loss metrics saved at ./loss_metrics{fold_number}.csv')
    args = parser.parse_args()

    main(args)

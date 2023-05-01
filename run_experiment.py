import argparse
import logging
import sys
import os
import copy
import time
import json

import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch import std_mean
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

from datasets import TgDataset, TmDataset, DensityDataset
from architectures.gcn import GCN
from architectures.mpnn import MPNN

def main(args):
    """Main experimental set-up for GCN configuration.

    Main method accepts pre-defined arguments in first parameter, including:
        - Etc.
    """
    
    lr = args.learning_rate
    wd = args.weight_decay
    n_convolutions = args.n_convolutions
    convolutions_dim = args.convolutions_dim
    n_hidden_layers = args.n_hidden_layers
    hidden_layers_dim = args.hidden_layers_dim
    batch_size = args.batch_size

    # Configure logging mechanism
    logging.basicConfig(stream=sys.stdout, format='%(message)s', level=logging.INFO if args.verbose else logging.CRITICAL)
    logger = logging.getLogger()
    
    # Dataset filepaths should generally remain the same
    save_path = '{save_dir}/'.format(save_dir=args.output_directory)
    raw_path = args.raw_path

    # Ensure path is writable
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Save run options
    json.dump(vars(args), open(os.path.join(save_path, f'config.json'), 'w'))
    
    # Assign seeds for deterministic random variables
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Select model architectures
    architecture_list = [GCN, MPNN]
    architecture_keys = ['gcn', 'mpnn']
    architecture_dict = dict(zip(architecture_keys, architecture_list))
    model_architecture = architecture_dict[args.model]
    
    # Select proper dataset
    dataset_list = [TgDataset, TmDataset, DensityDataset]
    dataset_keys = ['tg', 'tm', 'density']
    dataset_dict = dict(zip(dataset_keys, dataset_list))
    use_dataset = dataset_dict[args.target](root=raw_path)
    
    # Begin experiment
    nfolds = 5
    kfold = KFold(n_splits=nfolds, shuffle=True)

    for fold, (train_idx, test_idx) in tqdm(enumerate(kfold.split(use_dataset))):
        logger.info(f'\nFold {fold+1}/{nfolds}')

        train_set = use_dataset[train_idx.tolist()]
        test_set = use_dataset[test_idx.tolist()]

        y_set = torch.stack([dataobject.y for dataobject in train_set]) 
        std, mean = torch.std_mean(y_set, unbiased=False)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        model = model_architecture(n_convolutions, convolutions_dim, n_hidden_layers, hidden_layers_dim)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        # Early stopping variables
        patience = 100
        min_val_loss = math.inf
        epoch_since_min = 0
        min_model_coefficients = model.state_dict()

        # Training loop
        train_loss_list = []
        val_loss_list = []
        for epoch in range(5000):
            epoch_start = time.time_ns()
            logger.info(f'Epoch {epoch+1}')
            model.train()
            epoch_train_loss = 0
            for idx, data in enumerate(train_loader):
                data.y = (data.y-mean)/std
                data = data.to(device)
                endpoints = data.y.to(device)
                optimizer.zero_grad()
                outputs = model(data).view(-1)
                #outputs.require_grad = False
                loss = criterion(outputs, endpoints)
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            epoch_duration = time.time_ns() - epoch_start
            step_duration = epoch_duration / len(train_loader)
            logger.info(f'Epoch performance: {epoch_duration}ns ({step_duration}ns/step) ')
            epoch_train_loss /= len(train_loader)
            logger.info(f'Training Loss: {epoch_train_loss}')
            train_loss_list.append(epoch_train_loss)
            
            # Validation loss
            epoch_val_loss = 0
            model.eval()
            with torch.no_grad():
                for idx, data in enumerate(test_loader):
                    data.y = (data.y-mean)/std
                    data = data.to(device)
                    endpoints = data.y.to(device)
                    outputs = model(data).view(-1)
                    loss = criterion(outputs, endpoints)
                    epoch_val_loss += loss.item()
                epoch_val_loss /= len(test_loader)
                logger.info(f'Validation Loss: {epoch_val_loss}')
                val_loss_list.append(epoch_val_loss)
            
            # Early stopping mechanism to avoid overfit and save resources
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                epoch_since_min = 0
                min_model_coefficients = copy.deepcopy(model.state_dict())
            else:
                epoch_since_min += 1
            
            if epoch_since_min >= patience:
                model.load_state_dict(min_model_coefficients, strict=True)
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

        # Compute validation metrics
        y_true = []
        y_pred = []
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                data = data.to(device)
                endpoints = data.y.to(device)
                outputs = model(data).view(-1)
                y_true += endpoints.tolist()
                y_pred += ((outputs*std)+mean).tolist()

        MAE = mean_absolute_error(y_true, y_pred)
        MSE = mean_squared_error(y_true, y_pred)
        RMSE = mean_squared_error(y_true, y_pred, squared=False)
        R2 = r2_score(y_true, y_pred)

        metrics = pd.DataFrame({'MAE':[MAE], 'MSE':[MSE], 'RMSE':[RMSE], 'R2':[R2]})
        metrics = metrics.reset_index(drop=True)
        metrics.to_csv(os.path.join(save_path, f'metrics{str(fold+1)}.csv'), index=False)

        results = use_dataset.get_smiles(test_idx)
        results = results.reset_index(drop=True)
        results['true'] = y_true
        results['predicted'] = y_pred
        results.to_csv(os.path.join(save_path, f'results{str(fold+1)}.csv'), index=False)

        loss_metrics = pd.DataFrame({'Training Loss':train_loss_list, 'Validation Loss': val_loss_list})
        loss_metrics.to_csv(os.path.join(save_path, f'loss_metrics{str(fold+1)}.csv'), index=False)

        torch.save(model, os.path.join(save_path, f'fold{fold+1}.pt'))
        
        # Clear any memory (this is not required)
        del model, data
        with torch.no_grad():
            torch.cuda.empty_cache()


# command
# python ./run_experiment.py -t tm -rp ./data -o ./results -v True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'A python workflow for predicting polymer endpoints through a graph convolutional network.')    
    
    parser.add_argument('-t', '--target', type=str, required=True, help='Specify which endpoint dataset to run the model on. Options are tg, tm, or density.')
    parser.add_argument('-rp', '--raw_path', type=str, required=True, help='Specify path to raw data directory')
    parser.add_argument('-v', '--verbose', type=bool, help='If True, gives detailed status of model while running (eg. fold number, loss per epoch, time taken, etc). Default is False', default=False)
    parser.add_argument('-rs', '--random_seed', type=int, help='Initialize the random seed', default=0)
    parser.add_argument('-o', '--output_directory', type=str, required=True, help='Specify directory of folder to save at, containing models saved at ./fold{fold_number}.pt, results saved at ./results{fold_number}.csv, convenient performance metrics saved at ./metrics{fold_number}.csv, and loss metrics saved at ./loss_metrics{fold_number}.csv')
    parser.add_argument('-m', '--model', type=int, help='Model architecture (gcn, mpnn)', default='gcn', choices=['gcn','mpnn'])
    parser.add_argument('-lr', '--learning_rate', type=float, help='Set learning rate hyperparameter', default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, help='Set weight decay optimizer', default=1e-2)
    parser.add_argument('-bs', '--batch_size', type=int, help='Set batch size hyperparameter', default=128)
    parser.add_argument('-c', '--n_convolutions', type=int, help='Set the number of GCN layers', default=6)
    parser.add_argument('-cd', '--convolutions_dim', type=int, help='Set size of GCN layers', default=100)
    parser.add_argument('-f', '--n_hidden_layers', type=int, help='Set the number of hidden FC layers', default=2)
    parser.add_argument('-fd', '--hidden_layers_dim', type=int, help='Set size of FC layers', default=300)
    args = parser.parse_args()

    main(args)
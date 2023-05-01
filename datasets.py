import pandas as pd
import numpy as np

from rdkit import Chem

import torch
from torch_geometric.data import InMemoryDataset

from generate_graph_helpers import mol_to_graph

# class PolymerDataset(InMemoryDataset):
#     def process(self):
#         raw_data = pd.read_csv(self.raw_paths[0])

#         rd_mols = raw_data['SMILES'].apply(Chem.MolFromSmiles)
#         data_list = [mol_to_graph(mol) for mol in rd_mols]

#         for i, object in enumerate(data_list):
#             object.y = torch.tensor([raw_data.iloc[i, 1]], dtype=torch.float)

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

class TgDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['tg_raw.csv']

    @property
    def processed_file_names(self):
        return ['Tg_processed.pt']

    def download(self):
        #@TODO download CSV
        raise Exception('Download tg_raw.csv')

    def get_smiles(self, list_index):
        raw_data = pd.read_csv(self.raw_paths[0])
        if len(list_index) == 0:
            return raw_data[['SMILES']]
        return raw_data[['SMILES']].iloc[list_index]
    
    def process(self):
        raw_data = pd.read_csv(self.raw_paths[0])

        rd_mols = raw_data['SMILES'].apply(Chem.MolFromSmiles)
        data_list = [mol_to_graph(mol) for mol in rd_mols]

        for i, object in enumerate(data_list):
            object.y = torch.tensor([raw_data.iloc[i, 1]], dtype=torch.float)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DensityDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['density_raw.csv']

    @property
    def processed_file_names(self):
        return ['density_processed.pt']

    def download(self):
        #@TODO download CSV
        raise Exception('Download density_raw.csv')

    def get_smiles(self, list_index):
        raw_data = pd.read_csv(self.raw_paths[0])
        if len(list_index) == 0:
            return raw_data[['SMILES']]
        return raw_data[['SMILES']].iloc[list_index]

    def process(self):
        raw_data = pd.read_csv(self.raw_paths[0])

        rd_mols = raw_data['SMILES'].apply(Chem.MolFromSmiles)
        data_list = [mol_to_graph(mol) for mol in rd_mols]

        for i, object in enumerate(data_list):
            object.y = torch.tensor([raw_data.iloc[i, 1]], dtype=torch.float)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class TmDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['tm_raw.csv']

    @property
    def processed_file_names(self):
        return ['tm_processed.pt']

    def download(self):
        #@TODO download CSV
        raise Exception('Download tm_raw.csv')

    def get_smiles(self, list_index):
        raw_data = pd.read_csv(self.raw_paths[0])
        if len(list_index) == 0:
            return raw_data[['SMILES']]
        return raw_data[['SMILES']].iloc[list_index]

    def process(self):
        raw_data = pd.read_csv(self.raw_paths[0])

        rd_mols = raw_data['SMILES'].apply(Chem.MolFromSmiles)
        data_list = [mol_to_graph(mol) for mol in rd_mols]

        for i, object in enumerate(data_list):
            object.y = torch.tensor([raw_data.iloc[i, 1]], dtype=torch.float)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
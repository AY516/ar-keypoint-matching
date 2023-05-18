from datasets.base_dataset import BaseDataset
from utils.config import cfg
from pathlib import Path
import torch
from torch_geometric.datasets import WILLOWObjectClass as WILLOW
from torch_geometric.data import DataLoader

from utils import PairDataset


class WillowObject(BaseDataset):
    def __init__(self, transform):
        super(WillowObject, self).__init__()
        
        self.categories = WILLOW.categories
        self.root_path = Path(cfg.WILLOW.ROOT_DIR)

        self.train_len = cfg.WILLOW.TRAIN_NUM
        self.isValidation = cfg.WILLOW.VALIDATION
        self.transform = transform
    
    def load_dataset(self):
        datasets = [WILLOW(self.root_path, cat, self.transform) for cat in WILLOW.categories]
        datasets = [dataset.shuffle() for dataset in datasets]

        train_datasets = [dataset[:20] for dataset in datasets]
        train_datasets = [
            PairDataset(train_dataset, train_dataset, sample=False)
            for train_dataset in train_datasets
            ]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

        test_datasets = [dataset[20:] for dataset in datasets]


        test_datasets = [
            PairDataset(test_dataset, test_dataset, sample=False)
            for test_dataset in test_datasets
            ]
        # test_dataset = torch.utils.data.ConcatDataset(test_datasets)


        num_node_features = datasets[0].num_node_features
        num_edge_features = datasets[0].num_edge_features

        return train_dataset, test_datasets, num_node_features, num_edge_features
    
    def have_validation_set(self):
        return self.isValidation
        
         



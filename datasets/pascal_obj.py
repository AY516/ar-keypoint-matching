from datasets.base_dataset import BaseDataset
from utils.config import cfg
from pathlib import Path
import torch
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from utils import ValidPairDataset
from torch_geometric.data import DataLoader


class PascalObject(BaseDataset):
    def __init__(self, transform, pre_filter):
        super(PascalObject, self).__init__()

        self.root_path = Path(cfg.PASCALVOC.ROOT_DIR)
        self.transform = transform
        self.pre_filter = pre_filter
        self.isValidation = cfg.PASCALVOC.VALIDATION
        self.categories = PascalVOC.categories


    def load_dataset(self):
        train_datasets = []
        test_datasets = []
        for category in PascalVOC.categories:
            dataset = PascalVOC(self.root_path, category, train=True, transform=self.transform,
                                pre_filter=self.pre_filter)
            train_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
            dataset = PascalVOC(self.root_path, category, train=False, transform=self.transform,
                        pre_filter=self.pre_filter)
            test_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
        
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        num_node_features = dataset.num_node_features
        num_edge_features = dataset.num_edge_features

        return train_dataset, test_datasets, num_node_features,num_edge_features
    
    def have_validation_set(self):
        return self.isValidation
    
        


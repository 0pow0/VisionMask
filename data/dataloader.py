import os
import torch
import numpy as np

import pytorch_lightning as pl
import torchvision.transforms as T

from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path

from data.dataset import COCODataset, CUB200Dataset, MarioDataset, AtariDataset, DoomDataset, HighwayDataset

class HighwayDataModule(pl.LightningDataModule):
    def __init__(self, data_path, train_batch_size=15, val_batch_size=15, test_batch_size=15):
        super().__init__()
        self.data_path = data_path

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = HighwayDataset(self.data_path / 'train')
            self.val = HighwayDataset(self.data_path / 'val')

        if stage == "test" or stage is None:
            # self.test = HighwayDataset(self.data_path / 'test')
            self.test = HighwayDataset(self.data_path / 'all')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=self.collate_fn, 
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=self.collate_fn, 
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=self.collate_fn, 
                          num_workers=4)
    
    def collate_fn(self,batch):
        data = torch.stack([item[0] for item in batch])
        target = [item[1] for item in batch]
        return data, target

class DoomDataModule(pl.LightningDataModule):
    def __init__(self, data_path, train_batch_size=15, val_batch_size=15, test_batch_size=15):
        super().__init__()
        self.data_path = data_path

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = DoomDataset(self.data_path / 'train')
            self.val = DoomDataset(self.data_path / 'val')

        if stage == "test" or stage is None:
            self.test = DoomDataset(self.data_path / 'test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=self.collate_fn, 
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=self.collate_fn, 
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=self.collate_fn, 
                          num_workers=4)
    
    def collate_fn(self, batch):
        screens = torch.cat([item[0]['screen'] for item in batch])
        variables = torch.cat([item[0]['variables'] for item in batch])
        anno = [item[1] for item in batch]

        # Return a dictionary of batched tensors
        return {'screen': screens, 'variables': variables}, anno

class AtariDataModule(pl.LightningDataModule):
    def __init__(self, data_path, train_batch_size=15, val_batch_size=15, test_batch_size=15):
        super().__init__()
        self.data_path = data_path

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = AtariDataset(self.data_path / 'train')
            self.val = AtariDataset(self.data_path / 'val')

        if stage == "test" or stage is None:
            # self.test = AtariDataset(self.data_path / 'test')
            self.test = AtariDataset(self.data_path / 'one')
            # self.test = AtariDataset(self.data_path / 'all')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=self.collate_fn, 
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=self.collate_fn, 
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=self.collate_fn, 
                          num_workers=4)
    
    def collate_fn(self,batch):
        data = torch.stack([item[0] for item in batch])
        target = [item[1] for item in batch]
        return data, target

class MarioDataModule(pl.LightningDataModule):
    def __init__(self, data_path, train_batch_size=15, val_batch_size=15, test_batch_size=15, use_data_augmentation=False):
        super().__init__()
        self.data_path = data_path

        transformer = T.ToTensor()
        self.train_transformer = transformer
        self.test_transformer = transformer

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
    
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = MarioDataset(self.data_path / 'train', transform_fn=self.train_transformer)
            self.val = MarioDataset(self.data_path / 'val', transform_fn=self.train_transformer)

        if stage == "test" or stage is None:
            # self.test = MarioDataset(self.data_path / 'test', transform_fn=self.test_transformer)
            # self.test = MarioDataset(self.data_path / 'all', transform_fn=self.test_transformer)
            self.test = MarioDataset(self.data_path / 'one', transform_fn=self.test_transformer)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=self.collate_fn, 
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=self.collate_fn, 
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=self.collate_fn, 
                          num_workers=4)
    
    def collate_fn(self,batch):
        data = torch.stack([item[0] for item in batch])
        target = [item[1] for item in batch]
        return data, target

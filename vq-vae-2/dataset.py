import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb


CodeRow = namedtuple('CodeRow', ['label', 'top', 'bottom', 'filename'])


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename


class LMDBDataset(Dataset):
    def __init__(self, path, desired_class_label=None):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.valid_sample_indices = range(self.length)
        if desired_class_label is not None: self.filter_labels(desired_class_label)

    def filter_labels(self, desired_class_label):
        valid_sample_indices = []
        for i in range(self.length):
            label = self.__getitem__(i, only_label=True)
            if label == desired_class_label: valid_sample_indices.append(i)
        
        self.length = len(valid_sample_indices)        
        self.valid_sample_indices = valid_sample_indices
        
    def __len__(self):
        return self.length

    def __getitem__(self, index, only_label=False):
        valid_index = self.valid_sample_indices[index]
        with self.env.begin(write=False) as txn:
            key = str(valid_index).encode('utf-8')

            row = pickle.loads(txn.get(key))
            if only_label: return row.label
        return row.label, torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename

"""
Data loader for multi-view clustering datasets.

"""

import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Individual dataset classes
# ---------------------------------------------------------------------------

class Handwritten(Dataset):
    def __init__(self, path='./data/'):
        self.Y = scipy.io.loadmat(path + 'handwritten.mat')['Y'].astype(np.int32).reshape(2000,)
        self.V1 = scipy.io.loadmat(path + 'handwritten.mat')['X'][0][0].astype(np.float32)  #240
        self.V2 = scipy.io.loadmat(path + 'handwritten.mat')['X'][0][1].astype(np.float32)  #76


    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        x1 = self.normalize(self.V1[idx])
        x2 = self.normalize(self.V2[idx])
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    def normalize(self,x):
        """Normalize"""
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x

class Synthetic3D:
    def __init__(self, path=""):
        data = scipy.io.loadmat("./data/synthetic3d.mat")
        self.Y = data['Y'].astype(np.int32).reshape(600, )
        self.V1 = data['X'][0][0].astype(np.float32).reshape(600, -1)
        self.V2 = data['X'][1][0].astype(np.float32).reshape(600, -1)
        self.V3 = data['X'][2][0].astype(np.float32).reshape(600, -1)

    def __len__(self):
        return 600

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()





def load_data(dataset_name, data_dir='./data/'):
    """Load dataset by name. Returns (dataset, dims, view, data_size, class_num)."""
    dataset_config = {
        'Handwritten': (Handwritten, [240, 76], 2, 2000, 10),
        'Synthetic3D': (Synthetic3D, [3,3,3], 3, 600, 3)
    }

    if dataset_name not in dataset_config:
        raise NotImplementedError(f'Dataset "{dataset_name}" not supported. '
                                  f'Available: {list(dataset_config.keys())}')

    cls_or_fn, dims, view, data_size, class_num = dataset_config[dataset_name]
    if callable(cls_or_fn):
        dataset = cls_or_fn()
    else:
        dataset = cls_or_fn(data_dir)

    return dataset, dims, view, data_size, class_num

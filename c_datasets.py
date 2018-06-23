import os
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import pandas as pd
import numpy as np
import PIL.Image as Image
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pickle


class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.data = pd.read_csv("./data/train.csv")
        self.sorted_cls = sorted(list(set(self.data['Id'].tolist())))

        self.dictcls = OrderedDict()
        for i, cls in enumerate(self.sorted_cls):
            self.dictcls[i] = cls

        with open("testlist.txt", 'r') as f:
            self.names = f.read().splitlines()

    def __getitem__(self, index):

        img = Image.open('./data/test/'+self.names[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return self.names[index], img, 0

    def __len__(self):
        return len(self.names)


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        print(self.data.head())
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.transform = transform

        self.sorted_cls = sorted(list(set(self.data['Id'].tolist())))
        print(len(self.sorted_cls))
        print(self.sorted_cls[:10])

        self.clsdict = OrderedDict()
        for i, cls in enumerate(self.sorted_cls):
            self.clsdict[cls] = i

        self.dictcls = OrderedDict()
        for i, cls in enumerate(self.sorted_cls):
            self.dictcls[i] = cls

        # with open('clsdict.pkl', 'wb') as f:
        #     pickle.dump(self.clsdict, f)

        # with open('dictcls.pkl', 'wb') as f:
        #     pickle.dump(self.dictcls, f)
        # exit(0)

    def __getitem__(self, index):
        row = self.data.loc[[index]]
        imgpath, label = row.values.tolist()[0]
        cls_n = self.clsdict[label]

        img = Image.open('./data/train/'+imgpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return imgpath, img, cls_n

    def __len__(self):
        return len(self.data.index)

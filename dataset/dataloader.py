# -*- coding: UTF-8 -*-
from __future__ import division
import os
from torch.utils.data import Dataset
from PIL import Image


# 自定义数据集
class FrontierPredictionDataset(Dataset):
    """Frontier Prediction Dataset."""
    def __init__(self, txt_file, root_dir, transform=None):
        fh = open(txt_file, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append(words[0])
        self.imgs = imgs
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        x_name = os.path.join(self.root_dir, 'submaps', self.imgs[idx])
        y_name = os.path.join(self.root_dir, 'submaps_gt', self.imgs[idx])
        img_x = Image.open(x_name)
        img_y = Image.open(y_name)
        sample = {'image': img_x, 'image_gt': img_y}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.imgs)

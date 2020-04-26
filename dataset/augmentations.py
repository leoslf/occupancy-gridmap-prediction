# -*- coding: UTF-8 -*-
from __future__ import print_function, division
import os
import random
from PIL import Image
from collections import Iterable
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils


# 自定义transform
class Grayscale(object):
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, sample):
        img_x, img_y = sample['image'], sample['image_gt']
        img_x_out = F.to_grayscale(
            img_x, num_output_channels=self.num_output_channels)
        img_y_out = F.to_grayscale(
            img_y, num_output_channels=self.num_output_channels)
        return {'image': img_x_out, 'image_gt': img_y_out}

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(
            self.num_output_channels)


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable)
                                         and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        img_x, img_y = sample['image'], sample['image_gt']
        img_x_out = F.resize(img_x, self.size, self.interpolation)
        img_y_out = F.resize(img_y, self.size, self.interpolation)
        return {'image': img_x_out, 'image_gt': img_y_out}

    def __repr__(self):
        interpolate_str = pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, image_gt = sample['image'], sample['image_gt']
        return {
            'image': F.to_tensor(image),
            'image_gt': F.to_tensor(image_gt)
        }

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img_x, img_y = sample['image'], sample['image_gt']
        if random.random() < self.p:
            img_x_out = F.vflip(img_x)
            img_y_out = F.vflip(img_y)
        else:
            img_x_out = img_x
            img_y_out = img_y
        # return img_x,img_y
        return {'image': img_x_out, 'image_gt': img_y_out}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCenterCrop(object):
    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, sample):
        img_x, img_y = sample['image'], sample['image_gt']
        ratio = random.uniform(self.p, 1.0)
        size_x = int(img_x.size[0] * ratio)
        size_y = int(img_y.size[1] * ratio)
        img_x_out = F.center_crop(img_x, size_x)
        img_y_out = F.center_crop(img_y, size_y)
        return {'image': img_x_out, 'image_gt': img_y_out}

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class Normalize(object):
    def __init__(self, mean, std, mean_, std_, inplace=False):
        self.mean = mean
        self.std = std
        self.mean_ = mean_
        self.std_ = std_
        self.inplace = inplace

    def __call__(self, sample):
        img_x, img_y = sample['image'], sample['image_gt']
        img_x_out = F.normalize(img_x, self.mean, self.std, self.inplace)
        img_y_out = F.normalize(img_y, self.mean_, self.std_, self.inplace)

        return {'image': img_x_out, 'image_gt': img_y_out}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


aug_data_transform = transforms.Compose([
        RandomCenterCrop(),
        RandomVerticalFlip(),
        Resize((64, 64)),
        Grayscale(1),
        ToTensor()
        # Normalize(mean=(0.5,), std=(0.5,), mean_=(0.5,), std_=(0.5,))
    ])




# aug_data_transform = transforms.Compose([
#         Resize((32, 32)), 
#         Grayscale(1),
#         ToTensor(),
#         # Normalize(mean=(0.2695,), std=(0.4156,), mean_=(0.6181,), std_=(0.4011,))
#         # Normalize(mean=(0.5,), std=(0.5,), mean_=(0.5,), std_=(0.5,))
#     ])



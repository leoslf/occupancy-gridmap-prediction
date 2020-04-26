import numpy as np
import cv2
import random
import torch
import augmentations as aug
from dataloader import FrontierPredictionDataset

def get_pack_dir():
    import rospkg
    rospack = rospkg.RosPack()
    vae_map_prediction = rospack.get_path('vae_map_prediction')
    return vae_map_prediction

data_transform = aug.aug_data_transform

txt_file = get_pack_dir() + '/dataset_2/submaps/index.txt'
root_dir = get_pack_dir() + '/dataset_2'

dataset_frontiers = FrontierPredictionDataset(txt_file=txt_file,
                                                root_dir=root_dir,
                                                transform=data_transform)

train_loader = torch.utils.data.DataLoader(dataset=dataset_frontiers,
                                            batch_size=64,
                                            num_workers=1,
                                            shuffle=True)
# calculate means and std
mean = 0.
std = 0.
mean_ = 0.
std_ = 0.
nb_samples = 0.
nb_samples_ = 0.

for i, sample_batched in enumerate(train_loader):
    # measure data loading time
    input = sample_batched['image']
    target = sample_batched['image_gt']

    batch_samples = input.size(0)

    input = input.view(batch_samples, input.size(1), -1)
    mean += input.mean(2).sum(0)
    std += input.std(2).sum(0)
    nb_samples += batch_samples

    batch_samples_ = target.size(0)
    target = target.view(batch_samples, target.size(1), -1)
    mean_ += target.mean(2).sum(0)
    std_ += target.std(2).sum(0)
    nb_samples_ += batch_samples


mean /= nb_samples
std /= nb_samples
mean_ /= nb_samples_
std_ /= nb_samples_

print('mean: ', mean)
print('std: ', std)
print('mean_: ', mean_)
print('std_: ', std_)
import math
import numpy as np
import os
import torch

def get_pack_dir():
    import rospkg
    rospack = rospkg.RosPack()
    vae_map_prediction = rospack.get_path('vae_map_prediction')
    return vae_map_prediction

class DefaultConfigs(object):
    # String parameters
    # DataSet location
    txt_file = get_pack_dir() + '/frontiers_dataset/submaps/index.txt'
    root_dir = get_pack_dir() + '/frontiers_dataset'

    # Optimization options
    epochs = 450
    batch_size = 90
    learning_rate = 2e-4
    momentum = 0.9
    decay = 0.001 # Weight decay (L2 penalty)
    schedule = [180, 450] # Decrease learning rate at these epochs.
    gammas = [0.1, 0.01] # LR is multiplied by gamma on schedule, number of gammas should be equal to schedule

    # Checkpoints
    print_freq = 200
    save_path = './save' # Folder to save checkpoints and log.
    resume = '' # path to latest checkpoint (default: none)
    start_epoch = 0
    # evaluate = 

    # Acceleration
    num_gpu = 1
    use_cuda = num_gpu > 0 and torch.cuda.is_available()
    num_workers = 16

    # Random seed
    seed = 2020

config = DefaultConfigs()
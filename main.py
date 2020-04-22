# -*- coding: UTF-8 -*-
import os
import random
import time
import torch
import torchvision
import numpy as np
import pandas as pd
import warnings
import sys
from datetime import datetime
from torch import nn, optim
from config import config
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from timeit import default_timer as timer
from IPython import embed
from models.u_net import UNet
from models.residual_fully_conv_vae import ResidualFullyConvVAE
from dataset.dataloader import FrontierPredictionDataset
import utils
from torchvision import transforms
import dataset.augmentations as aug
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import cv2
from loss import loss_function


# 1. Set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

# 2. Define evaluate function
def evaluate(val_loader, model, criterion, epoch, log, writer):
    losses = utils.AverageMeter()
    accs = utils.AverageMeter()

    for i, sample_batched in enumerate(val_loader):
        # measure data loading time
        input = sample_batched['image']
        target = sample_batched['landmarks']

        if config.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda(async=True)

        # 计算输出
        output = model(input)
        #recon_batch, mu, logvariance = model(input)#VAE
        # 计算误差
        # loss = loss_function(output, target)
        loss = loss_function(output, target, loss_1='BCE')

        acc = 1-loss.item()

        # 传到 Tensorboard 可视化
        if epoch%20 == 0:
            output_ = vutils.make_grid(output, normalize=False, scale_each=True)
            images_ = vutils.make_grid(input, normalize=False, scale_each=True)
            label_ = vutils.make_grid(target, normalize=False, scale_each=True)
            writer.add_image('val/output_', output_, epoch)
            writer.add_image('val/images_', images_, epoch)
            writer.add_image('val/label_', label_, epoch)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        accs.update(acc, input.size(0))

    utils.print_log('    **Test** Prec@1 {accs:.3f} Error@1 {error1:.3f}'.format(accs=accs.avg, error1=losses.avg), log)
    return accs.avg, losses.avg


# 4. Define train function: forward, backward, update parameters
def train(train_loader, model, criterion, optimizer, epoch, log, writer):
    
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    accs = utils.AverageMeter()
    model.train()
    end = time.time()

    for i, sample_batched in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = sample_batched['image']
        target = sample_batched['landmarks']

        if config.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda(async=True)
        
        # 计算输出
        output = model(input)
        # 计算误差
        # loss = loss_function(output, target)
        loss = loss_function(output, target, loss_1='BCE')

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acc = 1-loss.item()
        accs.update(acc, input.size(0))

        # compute gradient and do SGD step
        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # 传到 Tensorboard 可视化
        if epoch%20 == 0:
            output_ = vutils.make_grid(output, normalize=False, scale_each=True)
            images_ = vutils.make_grid(input, normalize=False, scale_each=True)
            label_ = vutils.make_grid(target, normalize=False, scale_each=True)
            writer.add_image('train/output_', output_, epoch)
            writer.add_image('train/images_', images_, epoch)
            writer.add_image('train/label_', label_, epoch)
        
        # 打印训练状态
        if i % config.print_freq == 0:
            utils.print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                            'Loss {loss.val:.4f} ({loss.avg:.4f})   '.format(
                                epoch, i, len(train_loader), batch_time=batch_time,
                                data_time=data_time, loss=losses) + utils.time_string(), log)
    return accs.avg, losses.avg


# Main loop
def main():
    # Init logger
    writer = SummaryWriter()
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)
    log = open(os.path.join(config.save_path,
                            'log_seed_{}.txt'.format(config.seed)), 'w')
    utils.print_log('save path : {}'.format(config.save_path), log)
    utils.print_log("Random Seed: {}".format(config.seed), log)
    utils.print_log("python version : {}".format(
        sys.version.replace('\n', ' ')), log)
    utils.print_log("torch    version : {}".format(torch.__version__), log)
    utils.print_log("cudnn    version : {}".format(
        torch.backends.cudnn.version()), log)

    # Init dataset
    # data_transform= transforms.Compose([
    #     trans.RandomVerticalFlip(0.3),
    #     trans.RandomCenterCrop(0.75),
    #     trans.Resize((64, 64)),
    #     trans.Grayscale(1),
    #     trans.ToTensor()
    # ])
    data_transform = aug.aug_data_transform

    dataset_frontiers = FrontierPredictionDataset(txt_file=config.txt_file,
                                                  root_dir=config.root_dir,
                                                  transform=data_transform)

    train_data, val_data, test_data = utils.dataset_split(dataset_frontiers, 0.8)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=config.batch_size,
                                              num_workers=config.num_workers,
                                              shuffle=True)

    # Init model, criterion, and optimizer
    # utils.print_log("=> creating model '{}'".format(config.arch), log)
    # 4.2 get model and optimizer
    # model = get_net()
    net = UNet()
    #net = ResidualFullyConvVAE(64, latent_encoding_channels=64, skip_connection_type='concat')
    utils.print_log("=> network :\n {}".format(net), log)

    net = torch.nn.DataParallel(net, device_ids=list(range(config.num_gpu)))

    # define loss function (criterion) and optimizer
    
    #criterion = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.BCELoss()
    #criterion = torch.nn.L1Loss()
    # criterion = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), labels, reduction='sum')
    # optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
    #                             weight_decay=state['decay'], nesterov=True)
    optimizer = optim.Adam(net.parameters(),
                           lr=config.learning_rate,
                           amsgrad=True,
                           weight_decay=config.decay)

    if config.use_cuda:
        net.cuda()
        criterion.cuda()
        # criterion_L1.cuda()

    recorder = utils.RecorderMeter(config.epochs)

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            utils.print_log(
                "=> loading checkpoint '{}'".format(config.resume), log)
            checkpoint = torch.load(config.resume)
            recorder = checkpoint['recorder']
            config.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            utils.print_log("=> loaded checkpoint '{}' (epoch {})" .format(
                config.resume, checkpoint['epoch']), log)
        else:
            raise ValueError(
                "=> no checkpoint found at '{}'".format(config.resume))
    else:
        utils.print_log("=> do not use any checkpoint for this model", log)

    # if config.evaluate:
    #     evaluate(test_loader, net, criterion, log)
    #     return

    # Main loop
    start_time = time.time()
    epoch_time = utils.AverageMeter()
    for epoch in range(config.start_epoch, config.epochs):
        current_learning_rate = utils.adjust_learning_rate(
            optimizer, epoch, config.gammas, config.schedule)

        need_hour, need_mins, need_secs = utils.convert_secs2time(
            epoch_time.avg * (config.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        utils.print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.6f}]'.format(utils.time_string(), epoch, config.epochs, need_time, current_learning_rate)
                        + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)
        
        
        # train for one epoch
        train_acc, train_loss = train(train_loader, net, criterion, optimizer, epoch, log, writer)
        writer.add_scalar('Loss/Train_loss', train_loss, epoch)
        
        # eval every 25 epoch
        if epoch%25==0:
            val_acc, val_loss = evaluate(test_loader, net, criterion, epoch, log, writer)
            writer.add_scalar('Loss/Val_loss', val_loss, epoch)
            #writer.add_scalar('Loss', {'Train Loss': train_loss,'Val Loss': val_loss}, epoch)

        
        is_best = recorder.update(
            epoch, train_loss, train_acc, val_loss, val_acc)
        
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, is_best, config.save_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        #recorder.plot_curve(os.path.join(config.save_path, 'curve.png'))

        if epoch == 250:
            torch.save(net, '250_model.pth')  
        if epoch == 499:
            torch.save(net, '499_model.pth')  
        writer.close()
    log.close()


if __name__ == "__main__":
    #BCE
    # config.batch_size = 128
    # config.learning_rate = 1e-3
    # config.epochs = 800
    # main()
    # config.learning_rate = 1e-4
    # main()
    # config.learning_rate = 1e-5
    # main()
    # config.learning_rate = 1e-6
    # main()

    #BCE
    # config.epochs = 500
    # config.batch_size = 16
    # config.learning_rate = 1e-2
    # main()
    # config.learning_rate = 1e-3
    # main()
    # config.batch_size = 64
    # config.learning_rate = 1e-2
    # main() #19-24
    # config.learning_rate = 1e-3
    # main()

    #BCE+ssim
    #0.7*loss + 0.3*SSIM
    # config.epochs = 500
    # config.batch_size = 16
    # config.learning_rate = 1e-3
    # main()
    # config.learning_rate = 1e-4
    # main()
    # config.batch_size = 64
    # config.learning_rate = 1e-3
    # main()
    # config.learning_rate = 1e-4
    # main()
    
    #BCE+ssim
    #0.8*loss + 0.2*SSIM
    # config.epochs = 300
    # config.batch_size = 8
    # config.learning_rate = 1e-3
    # main()
    # config.batch_size = 16
    # config.learning_rate = 1e-3
    # main()
    # config.epochs = 500
    # config.batch_size = 32
    # config.learning_rate = 1e-3
    # main()


    #0.8*loss + 0.2*SSIM
    # config.epochs = 400
    # config.batch_size = 32
    # config.learning_rate = 1e-3
    # schedule = [180, 400] # Decrease learning rate at these epochs.
    # gammas = [0.8, 0.1] # LR is multiplied by gamma on schedule, number of gammas should be equal to schedule
    # main()

    #0.8*loss + 0.2*SSIM
    # config.epochs = 1000
    # config.batch_size = 64
    # config.learning_rate = 1e-4
    # schedule = [500, 1000] # Decrease learning rate at these epochs.
    # gammas = [0.8, 0.1] # LR is multiplied by gamma on schedule, number of gammas should be equal to schedule
    # main()

    #Feb11 17 37
    # BCE
    config.epochs = 500
    config.batch_size = 64
    config.learning_rate = 1e-4
    schedule = [250, 500] # Decrease learning rate at these epochs.
    gammas = [0.8, 0.1] # LR is multiplied by gamma on schedule, number of gammas should be equal to schedule
    main()
# -*- coding: UTF-8 -*-
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from torch import optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import dataset.augmentations as aug
import utils
from dataset.dataloader import FrontierPredictionDataset
from loss import loss_function

from models.u_net import UNet

# 1. Set random.seed and cudnn performance
config = utils.Config('config.yml')
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
        x = sample_batched['image']
        target = sample_batched['image_gt']

        if config.use_cuda:
            target = target.cuda(async=True)
            x = x.cuda(async=True)

        # feed in model
        y = model(x)
        loss = loss_function(y, target, loss_1='BCE')

        acc = 1 - loss.item()

        # visualization in Tensorboard
        if epoch % config.vis_step == 0:
            y_ = vutils.make_grid(y, normalize=False, scale_each=True)
            images_ = vutils.make_grid(x, normalize=False, scale_each=True)
            label_ = vutils.make_grid(target, normalize=False, scale_each=True)
            writer.add_image('val/y_', y_, epoch)
            writer.add_image('val/images_', images_, epoch)
            writer.add_image('val/label_', label_, epoch)

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))
        accs.update(acc, x.size(0))

    utils.print_log(
        '    **Test** Prec@1 {accs:.3f} Error@1 {error1:.3f}'.format(
            accs=accs.avg, error1=losses.avg), log)
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
        x = sample_batched['image']
        target = sample_batched['image_gt']

        if config.use_cuda:
            target = target.cuda(async=True)
            x = x.cuda(async=True)

        y = model(x)

        loss = loss_function(
            y,
            target,
            loss_1='BCE',
        )

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))
        # acc = 1 - loss.item()
        # accs.update(acc, x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # visualize in tensorboard
        if epoch % 20 == 0:
            y_ = vutils.make_grid(y, normalize=False, scale_each=True)
            images_ = vutils.make_grid(x, normalize=False, scale_each=True)
            label_ = vutils.make_grid(target, normalize=False, scale_each=True)
            writer.add_image('train/y_', y_, epoch)
            writer.add_image('train/images_', images_, epoch)
            writer.add_image('train/label_', label_, epoch)

        if i % config.print_freq == 0:
            utils.print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses) + utils.time_string(), log)
    return accs.avg, losses.avg


# Main loop
def main():
    # Init logger
    writer = SummaryWriter()
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)
    log = open(
        os.path.join(config.save_path, 'log_seed_{}.txt'.format(config.seed)),
        'w')
    utils.print_log('save path : {}'.format(config.save_path), log)
    utils.print_log("Random Seed: {}".format(config.seed), log)
    utils.print_log(
        "python version : {}".format(sys.version.replace('\n', ' ')), log)
    utils.print_log("torch    version : {}".format(torch.__version__), log)
    utils.print_log(
        "cudnn    version : {}".format(torch.backends.cudnn.version()), log)

    data_transform = aug.aug_data_transform

    dataset_frontiers = FrontierPredictionDataset(txt_file=config.txt_file,
                                                  root_dir=config.root_dir,
                                                  transform=data_transform)

    train_data, val_data, test_data = utils.dataset_split(
        dataset_frontiers, 0.8)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=config.batch_size,
                                              num_workers=config.num_workers,
                                              shuffle=True)

    # Init model, criterion, and optimizer
    net = UNet()
    net = torch.nn.DataParallel(net, device_ids=list(range(config.num_gpu)))
    writer.add_graph(net, torch.rand(size=(8, 1, 64, 64)))

    optimizer = optim.Adam(net.parameters(),
                           lr=config.learning_rate,
                           amsgrad=True,
                           weight_decay=config.decay)
    
    # this criterion has no effect on loss
    # another loss function with ssim is used
    criterion = torch.nn.BCELoss() 


    if config.use_cuda:
        net.cuda()
        criterion.cuda()
        # criterion_L1.cuda()

    recorder = utils.RecorderMeter(config.epochs)

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            utils.print_log("=> loading checkpoint '{}'".format(config.resume),
                            log)
            checkpoint = torch.load(config.resume)
            recorder = checkpoint['recorder']
            config.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            utils.print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    config.resume, checkpoint['epoch']), log)
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(
                config.resume))
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
            config.learning_rate, optimizer, epoch, config.gammas,
            config.schedule)

        need_hour, need_mins, need_secs = utils.convert_secs2time(
            epoch_time.avg * (config.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        utils.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.6f}]'.
            format(utils.time_string(), epoch, config.epochs, need_time,
                   current_learning_rate) +
            ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(
                recorder.max_accuracy(False),
                100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_loss = train(train_loader, net, criterion, optimizer,
                                      epoch, log, writer)
        writer.add_scalar('Loss/Train_loss', train_loss, epoch)

        # eval every val_step epoch
        if epoch % config.val_step == 0:
            val_acc, val_loss = evaluate(test_loader, net, criterion, epoch,
                                         log, writer)
            writer.add_scalar('Loss/Val_loss', val_loss, epoch)
            #writer.add_scalar('Loss', {'Train Loss': train_loss,'Val Loss': val_loss}, epoch)

        is_best = recorder.update(epoch, train_loss, train_acc, val_loss,
                                  val_acc)

        utils.save_checkpoint(
            {
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
    main()

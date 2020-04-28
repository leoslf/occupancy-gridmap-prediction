# -*- coding: UTF-8 -*-
import sys
import os
import random
import time
import warnings
import logging
from logging import FileHandler, StreamHandler, Formatter
from functools import *
from itertools import *

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import dataset.augmentations as aug
import utils
from dataset.dataloader import FrontierPredictionDataset
from loss import loss_function

class BaseModel(object):
    def __init__(self, log_level = logging.INFO, *argv, **kwargs):
        self.log_level = log_level
        self.argv = argv
        self.__dict__.update(kwargs)
        # Common Initializations
        self.common_init()

        # Model specific / convenience initialization
        self.init()


        # Load Dataset
        self.load_dataset()
        # Decide whether to use CPU/GPU
        self.select_processing_unit()


        # [Optional] Load checkpoint
        # NOTE: condition handled inside
        self.resuming()

    @property
    def name(self):
        """ Automatically using the class name of the subclass """
        return self.__class__.__name__
    
    def common_init(self):
        self.config = self.prepare_config()
        # Setup environment
        self.environment_setup()

        # Prepare model
        self.model = self.prepare_model()
        if self.config.use_cuda:
            self.model = self.model.cuda()
        self.net = torch.nn.DataParallel(self.model, device_ids=list(range(self.config.num_gpu)))
        self.optimizer = self.prepare_optimizer()

        # Setup logging facilities
        self.logging_setup()
        # Output misc info: e.g. versions
        self.log_misc_info()

    @property
    def config_filename(self):
        return "config.yml"

    def prepare_config(self):
        return utils.Config(self.config_filename)

    def init(self):
        """
        Initialization 

        Just to avoid writing constructor over and over when subclassing
        """
        pass

    def environment_setup(self):
        """ Wrapper for Environment Setup """
        self.config_random()
        self.config_cudnn()
        self.config_suppression()

    def config_random(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

    def config_cudnn(self):
        torch.backends.cudnn.benchmark = True

    def config_suppression(self):
        warnings.filterwarnings("ignore")

    @property
    def log_filename(self):
        return os.path.join(self.config.save_path, "log_seed_%s.txt" % self.config.seed)

    @property
    def log_format(self):
        return "[%(asctime)s] %(name)s [%(levelname)s] (%(filename)s:%(funcName)s:%(lineno)d): %(message)s"

    @property
    def log_stream(self):
        return sys.stdout

    @property
    def dummy_input(self):
        input = torch.rand(size=(8, 1, 64, 64))
        if self.config.use_cuda:
            input = input.cuda(async=True)
        return input

    def logging_setup(self):
        """ Setup Common Logging Facilities """
        logging.basicConfig(level=self.log_level, format=self.log_format)
        # Creating logger with subclass's class name
        self.logger = logging.getLogger(self.name)
        # TODO: refactor this into file-based config, this is ugly
        # log_formatter = Formatter(self.log_format)
        # Use Local time instead of UTC
        # log_formatter.converter = time.localtime
        def add_handler(handler):
            # handler.setFormatter(log_formatter)
            self.logger.addHandler(handler)

        handlers = [
            # File output
            FileHandler(self.log_filename),
            # Stream output
            # StreamHandler(self.log_stream),
        ]

        for handler in handlers:
            add_handler(handler)

        # Summary Writer for Tensorboard
        self.writer = SummaryWriter()
        # TODO: Attempt to remove error
        # self.writer.add_graph(self.net.module, self.dummy_input)
        self.writer.add_graph(self.net, self.dummy_input)

    def log_misc_info(self):
        misc_info = {
            "Save path": self.config.save_path,
            "Random seed": self.config.seed,
            "Python version": sys.version.replace("\n", " "),
            "Torch version": torch.__version__,
            "cudnn version": torch.backends.cudnn.version(),
        }

        for (displayed_name, value) in misc_info.items():
            self.logger.info("%s: %r", displayed_name, value)

    @property
    def dataloader_kwargs(self):
        return {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "shuffle": True
        }

    def load_dataset(self):
        """ Common procedures to load dataset """
        dataset_frontiers = FrontierPredictionDataset(txt_file=self.config.txt_file,
                                                      root_dir=self.config.root_dir,
                                                      transform=aug.aug_data_transform)

        # Train-val-Test split
        self.train_data, self.val_data, self.test_data = utils.dataset_split(dataset_frontiers, self.dataset_split_ratio)

        self.train_loader = DataLoader(self.train_data, **self.dataloader_kwargs)
        self.val_loader = DataLoader(self.val_data, **self.dataloader_kwargs)
        self.test_loader  = DataLoader(self.test_data, **self.dataloader_kwargs)

    @property
    def dataset_split_ratio(self):
        raise NotImplementedError("Split ratio must be overrided by subclass")
    
    def prepare_model(self):
        raise NotImplementedError("Model must be overrided by subclass")

    def prepare_optimizer(self):
        return optim.Adam(self.net.parameters(),
                          lr=self.config.learning_rate,
                          amsgrad=True,
                          weight_decay=self.config.decay)
    @property
    def criterion(self):
        """
        this criterion has no effect on loss
        another loss function with ssim is used
        """
        return torch.nn.BCELoss() 

    def select_processing_unit(self):
        if self.config.use_cuda:
            self.net.cuda()
            self.criterion.cuda()
            # self.criterion_L1.cuda()

    @property
    def recorder(self):
        return utils.RecorderMeter(self.config.epochs)

    def resuming(self):
        if self.config.resume:
            self.load_checkpoint(self.config.resume)
            return
        # Not loading checkpoint
        self.logger.info("Not resuming with any checkpoints: config.resume evaluated as falsy value")

    def main_loop(self):

        start_time = time.time()
        epoch_time = utils.AverageMeter()

        for epoch in range(self.config.start_epoch, self.config.epochs):
            # Current learning rate
            learning_rate = utils.adjust_learning_rate(self.config.learning_rate,
                                                       self.optimizer,
                                                       epoch,
                                                       self.config.gammas,
                                                       self.config.schedule)

            self.logger.info("%s [Epoch=%03d/%03d] %s [learning_rate=%6.6f] [Best : Accuracy%.2f, Error%.2f]", utils.time_string(), epoch, self.config.epochs, utils.secs2time_string(epoch_time.avg * (self.config.epochs - epoch)), learning_rate, self.recorder.max_accuracy(False), 100 - self.recorder.max_accuracy(False))

            # Train for one epoch
            train_acc, train_loss = self.train(epoch)
            self.writer.add_scalar("Loss/Train_loss", train_loss, epoch)

            # eval every val_step epoch
            if epoch % self.config.val_step == 0:
                val_acc, val_loss = self.evaluate(epoch)
                self.writer.add_scalar("Loss/Val_loss", val_loss, epoch)
                # self.writer.add_scalar("Loss", {"Train Loss": train_loss,"Val Loss": val_loss}, epoch)

            is_best = self.recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)

            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.net.state_dict(),
                    "recorder": self.recorder,
                    "optimizer": self.optimizer.state_dict(),
                }, is_best, self.config.save_path, "checkpoint.pth.tar")

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()
            # self.recorder.plot_curve(os.path.join(self.config.save_path, 'curve.png'))

            if epoch == 250:
                torch.save(self.net, '250_model.pth')
            if epoch == 499:
                torch.save(self.net, '499_model.pth')

            # FIXME: 
            # flushing instead of closing
            # self.writer.flush()

    @property
    def X_key(self):
        return "image"

    @property
    def GT_key(self):
        return "image_gt"

    def loss_function(self, x, ground_truth):
        """ Wrapper/Adapter of the loss function (Model Specific)

        This encapsulates the variables other than output, and feed them into the loss function.
        
        Returns:
            (prediction, loss)
        """
        raise NotImplementedError("handle_loss must be overrided by subclass")

    def visualize(self, prefix, epoch, prediction, images, label):
        with torch.no_grad():
            keys = ["prediction_", "images_", "labels_"]
            make_grid = partial(vutils.make_grid, normalize=False, scale_each=True)
            for (key, value) in zip(keys, map(make_grid, (prediction, images, label))):
                self.writer.add_image("%s/%s" % (prefix, key), value, epoch)

    def train(self, epoch):
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        accs = utils.AverageMeter()

        self.model.train()
        end = time.time()

        for batch_no, sample_batched in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            x = sample_batched[self.X_key]
            ground_truth = sample_batched["image_gt"]

            if self.config.use_cuda:
                x = x.cuda(async=True)
                ground_truth = ground_truth.cuda(async=True)

            # Only output tuple (prediction, loss)
            prediction, loss = self.loss_function(x, ground_truth)

            # measure accuracy and record loss
            losses.update(loss.item(), x.size(0))
            acc = 1 - loss.item()
            accs.update(acc, x.size(0))

            # compute gradient and do SGD step
            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            # visualize in tensorboard
            if epoch % self.config.vis_step == 0:
                self.visualize("train", epoch, prediction, x, ground_truth)

            # Outputs the Training status
            if batch_no % self.config.print_freq == 0:
                self.logger.info(
                    "  [Epoch: {:03d}][Batch: {:03d}/{:03d}]   "
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})   "
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})   "
                    "Loss {loss.val:.4f} ({loss.avg:.4f})   ".format(
                        epoch,
                        batch_no + 1,
                        len(self.train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses))

        return accs.avg, losses.avg

    def evaluate(self, epoch):
        losses = utils.AverageMeter()
        accs = utils.AverageMeter()

        for i, sample_batched in enumerate(self.val_loader):
            # measure data loading time
            x = sample_batched[self.X_key]
            ground_truth = sample_batched[self.GT_key]

            if self.config.use_cuda:
                x = x.cuda(async=True)
                ground_truth = ground_truth.cuda(async=True)

            prediction, loss = self.loss_function(x, ground_truth)

            acc = 1 - loss.item()

            # visualize in tensorboard
            if epoch % self.config.vis_step == 0:
                self.visualize("val", epoch, prediction, x, ground_truth)

            # measure accuracy and record loss
            losses.update(loss.item(), x.size(0))
            accs.update(acc, x.size(0))

        self.logger.info("    **Test** Prec@1 {accs:.3f} Error@1 {error1:.3f}".format(accs=accs.avg, error1=losses.avg))

        return accs.avg, losses.avg



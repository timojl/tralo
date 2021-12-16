import os
import time
import copy
from datetime import datetime
from functools import partial
from itertools import combinations
from os.path import join, isfile, isdir, expanduser
import matplotlib.pyplot as plt


import json
import math
import inspect
import numpy as np
import pandas as pd
import torch
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as nnf
from torch import nn
from torch.optim import SGD, AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from torchvision import transforms

from tralo.training_tracker import TrainingTracker
from tralo.log import log
from tralo.utils import AttributeDict
from tralo.experiments import get_attribute, filter_args, load_model


def cosine_warmup_lr(i, warmup=10, max_iter=90):
    """ Cosine LR with Warmup """
    if i < warmup:
        return (i+1)/(warmup+1)
    else:
        return 0.5 + 0.5*math.cos(math.pi*(((i-warmup)/(max_iter- warmup))))


def score(cfg, train_checkpoint_id, train_cfg):
    """ Function to compute test scores 
        cfg: test configuration (test_configuration in cifar_example.yaml)
        train_checkpoint_id: name of the trained model
        train_cfg: training configuration
    """

    from torchvision.datasets import CIFAR10

    # cfg is the test configuration, # train_config contains the original training configuration

    cfg = AttributeDict(cfg)
    train_cfg = AttributeDict(train_cfg)
    
    # the training config can also be obtained from disk
    # train_config_disk = AttributeDict(json.load(open(f'logs/{train_checkpoint_id}/config.json')))

    if train_cfg.load_model is not None:
        model_cls = get_attribute(train_cfg.load_model)
        _, model_args, _ = filter_args(train_cfg, inspect.signature(model_cls).parameters)
        model = model_cls(**model_args)
    else:
        weights_file = 'weights.pth' if cfg.use_weights is None else cfg.use_weights
        model_cls = get_attribute(train_cfg.model)
        model = model_cls(num_classes=10)
        model.load_state_dict(torch.load(join('logs', train_checkpoint_id, 'weights.pth')))
    
    model.cuda()
    model.eval()
    device = next(model.parameters()).device

    my_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CIFAR10(root='~/datasets/Cifar10', train=False, transform=my_transforms)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, drop_last=False)

    losses = []
    hits = []
    with torch.no_grad():
        for i, (data_x, data_y) in enumerate(loader):

            pred = model(data_x.to(device))

            hits += (pred.argmax(1) == data_y.to(device)).cpu().tolist()

            if i > 10:
                break

    return [['acc', np.mean(hits)]]


def val_callback(model):
    """ Callback for validation """

    from torchvision.datasets import CIFAR10

    my_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    model.eval()

    device = next(model.parameters()).device

    dataset = CIFAR10(root='~/datasets/Cifar10', transform=my_transforms)
    dataset.train_list[:1]

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, drop_last=False)

    losses = []
    hits = []
    with torch.no_grad():
        for data_x, data_y in loader:

            pred = model(data_x.to(device))

            hits += (pred.argmax(1) == data_y.to(device)).cpu().tolist()

    return [['acc', np.mean(hits)]]
            


def train(cfg):

    # cfg contains the variables defined in cifar_example.yaml

    from torchvision.datasets import CIFAR10

    if cfg.load_model is not None:
        return None

    workers_suggested = len(os.sched_getaffinity(0))
    workers = min(workers_suggested, cfg.workers) if 'workers' in cfg else workers_suggested

    my_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # dynamic way to initialize dataset using matching arguments from the config
    # dataset_cls = get_attribute(cfg.dataset)
    #  _, dataset_args, _ = filter_args(train_config, inspect.signature(dataset_cls).parameters)
    # dataset = dataset_cls(**dataset_args)

    dataset = CIFAR10(root='~/datasets/Cifar10', transform=my_transforms)
    dataset.train_list[1:]

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=workers,
                        pin_memory=cfg.pin)

    # dynamic initialization using model specified in config
    model_cls = get_attribute(cfg.model)
    model = model_cls(num_classes=10).cuda()
    device = next(model.parameters()).device

    if cfg.init_weights:    
        model.load_state_dict(torch.load(expanduser(cfg.load_weights)), strict=True)

    filename = cfg.name

    log.info(f'OPTIMIZATION: {cfg.optimizer}, lr:{cfg.lr} BS:{cfg.batch_size}, lr_scheduler: {cfg.lr_scheduler}')

    opt_cls = get_attribute(cfg.optimizer)
    opt_args = {} if cfg.optimizer_args is None else cfg.optimizer_args
    opt = opt_cls(model.parameters(), lr=cfg.lr, **opt_args)

    if cfg.lr_scheduler == 'cosine_warmup':
        lr_scheduler = LambdaLR(opt, partial(cosine_warmup_lr, max_iter=(cfg.max_iterations), warmup=cfg.warmup))
    else:
        lr_scheduler = None

    loss_fn = get_attribute(cfg.loss)

    if cfg.amp:
        log.info('Using AMP')
        autocast_fn = autocast
        scaler = GradScaler()
    else:
        autocast_fn, scaler = nullcontext, None

    i_iter = 0

    with TrainingTracker(filename, model=model, config=cfg, interval=cfg.interval, 
                        metric=(val_callback, cfg.val_interval),
                        grad_weights=['cnn.conv1.weight', 'cnn.conv1.weight']) as logger:

        cfg.assume_no_unused_keys(exceptions=['tau', 'checkpoint_iterations'])        

        log.info(f'start training: {filename}')
        log.info(f'#iterations: {cfg.max_iterations}, workers: {cfg.workers}')

        try:
            while i_iter < cfg.max_iterations:

                for _, (data_x, data_y) in enumerate(loader):

                    model.train()

                    data_x, data_y = data_x.to(device), data_y.to(device)
                
                    opt.zero_grad()

                    pred = model(data_x)

                    with autocast_fn():
                        loss = loss_fn(pred, data_y)

                    if scaler is None:
                        loss.backward()
                        opt.step()
                    else:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()

                    logger.iter(i_iter, loss)

                    # one step per iteration
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    if torch.isnan(loss).item() or torch.isinf(loss).item():
                        log.info('Stopping because of nan/inf loss')
                        break

                    i_iter += 1
                    if i_iter > cfg.max_iterations:
                        break

            # final val
            val_callback(model)

        except KeyboardInterrupt:
            log.info('Stopped by keyboard. Collecting processes...')

        log.info('Training complete')

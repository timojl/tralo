from tralo.experiments import get_attribute, load_model, AttributeDict

from torchvision.datasets import MNIST
import torch
import inspect
from tralo.utils import filter_args
from tralo import TrainingTracker
import random
import time
import numpy as np
from os.path import expanduser, join

from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as nnf

from torchvision import transforms


class Model(nn.Module):

    def __init__(self, k=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8*k, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(8*k, k * 8, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(k * 8, k*16, 3),
            nn.MaxPool2d(3),
            nn.Conv2d(k*16, 10, 1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)

tf = transforms.Compose([
    transforms.ToTensor(),
])

def train_loop(config):

    config = AttributeDict(config)

    model_cls = get_attribute(config.model)
    _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)
    model = model_cls(**model_args)

    if config.cuda:
        model.cuda()
    
    device = next(model.parameters()).device
 
    dataset = MNIST(expanduser('~/datasets'), transform=tf, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)

    opt_cls = get_attribute(config.optimizer)
    opt = opt_cls(model.parameters(), lr=config.lr)

    loss_fn = get_attribute(config.loss)

    with TrainingTracker(log_dir=config.name, model=model, config=config, fixed_iterations=(2, 5, 10)) as log:

        i = 0
        for i_epoch in range(config.max_epochs):
            for data_x, data_y in data_loader:

                data_x, data_y = data_x.to(device), data_y.to(device)

                pred = model(data_x)
                loss = loss_fn(pred, data_y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                log.iter(i=i, loss=loss)
                i += 1
                
                if i >= config.max_iterations:
                    log.save_weights()
                    return model

        log.save_weights()
        return model


def score(config, train_checkpoint_id):

    dataset = MNIST(expanduser('~/datasets'), transform=tf, download=True, train=False)

    loss_fn = get_attribute(config.loss)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)

    model = load_model(train_checkpoint_id)

    if config.cuda:
        model.cuda()
        
    device = next(model.parameters()).device

    model.eval()

    i, losses, hits = 0, [], []
    for data_x, data_y in data_loader:

        data_x, data_y = data_x.to(device), data_y.to(device)

        pred = model(data_x)
        loss = loss_fn(pred, data_y)

        losses += [float(loss)]
        hits += (pred.argmax(1) == data_y).tolist()

        i += 1
        if i >= config.max_iterations:
            break
        
    scores = dict(mnist=dict(accuracy=np.mean(hits), loss=np.mean(losses), n_samples=len(losses)))
    return scores


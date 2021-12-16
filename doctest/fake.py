from torchvision.datasets import MNIST
import torch
from tralo import TrainingTracker
import random
import time
import numpy as np
from os.path import expanduser

from torch import nn
from torch.utils.data import DataLoader

configs = [
    dict(a=1, b='Muh'),
]

for i in range(len(configs)):

    with TrainingTracker(f'run_{i}', config=configs[i]) as log:

        for i_epoch in range(10):
            for i_iter in range(100):

                fake_loss = np.random.normal() + 3 - (i_epoch*100 + i_iter) / 1000
                log.iter(i_epoch*100 + i_iter, loss=fake_loss)

                pass

            # let's assume we evaluate the metric only once per epoch
            fake_metric = np.random.normal() + (i_epoch*100 + i_iter) / 1000
            log.iter(i_epoch*100 + i_iter, fake_metric=fake_metric)

        print(log.metric_values)

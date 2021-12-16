from torchvision.datasets import MNIST
from tralo import TrainingTracker
from os.path import expanduser

import torch
from torch import nn
from torch.nn import functional as nnf


from torchvision import transforms

tf = transforms.Compose([
    transforms.ToTensor(),
])

mnist = MNIST(expanduser('~/datasets'), transform=tf, download=True)

data_loader = torch.utils.data.DataLoader(mnist, batch_size=64)

model = nn.Sequential(
    nn.Conv2d(1, 4, 3),
    nn.MaxPool2d(2),
    nn.Conv2d(4, 8, 3),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3),
    nn.MaxPool2d(3),
    nn.Conv2d(16, 10, 1),
    nn.Flatten()
)

opt = torch.optim.Adam(model.parameters(), lr=0.001)

with TrainingTracker(f'mnist_run', interval=10, utility=True, gradient_weights=[model[2]]) as log:
    i = 0
    for i_epoch in range(1):
        for data_x, data_y in data_loader:

            pred = model(data_x)
            loss = nnf.cross_entropy(pred, data_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            log.iter(i=i, loss=loss)
            i += 1
            
            if i > 100:
                break


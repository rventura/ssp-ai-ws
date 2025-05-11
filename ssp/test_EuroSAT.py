#!/usr/bin/env python3

import sys
import torch
from torch import nn
from torch.utils.data import *
from torchvision import datasets
from torchvision.transforms import *
from torchvision.io import decode_image
import time


# Download EuroSAT dataset
full_dataset = datasets.EuroSAT(
    root="data",
    download=True,
    transform=ToTensor(),
)

dataset_size = len(full_dataset)

# Create data loaders.
full_dataloader      = DataLoader(full_dataset)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
import torch.nn.functional as F

class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(96 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm2d(48)
        self.batchnorm2 = nn.BatchNorm2d(48)
        self.batchnorm3 = nn.BatchNorm2d(96)
        self.batchnorm4 = nn.BatchNorm2d(96)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(F.relu(self.batchnorm4(self.conv4(x))))
        x = x.view(-1, 96 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = BigCNN().to(device)
model.load_state_dict(torch.load(sys.argv[1], weights_only=True))

loss_fn = nn.CrossEntropyLoss()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct


test(full_dataloader, model, loss_fn)

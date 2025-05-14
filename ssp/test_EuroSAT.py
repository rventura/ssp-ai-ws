#!/usr/bin/env python3

import sys
import torch
from torch import nn
from torch.utils.data import *
from torchvision import datasets
from torchvision.transforms import *
from torchvision.io import decode_image
import time

from models_EuroSAT import *


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


model = VGG16().to(device)
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

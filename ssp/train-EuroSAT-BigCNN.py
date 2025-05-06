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

# Split the dataset
train_dataset, validation_dataset, test_dataset = random_split(
    full_dataset, [int(0.7*dataset_size),
                   int(0.15*dataset_size),
                   int(0.15*dataset_size)]
)

LABELS = ["A.Crop", "Forest", "Vegetation", "Highway", "Industrial", "Pasture", "P.Crop", "Residential", "River", "Sea/Lake"]

batch_size = int(sys.argv[2])

# Create data loaders.
train_dataloader      = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
test_dataloader       = DataLoader(test_dataset, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

torch.accelerator.set_device_index(int(sys.argv[1]))

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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if True: # batch % 20 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# DELME
def train_timed(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    t0 = time.monotonic()
    for batch, (X, y) in enumerate(dataloader):
        print(f"0: {time.monotonic()-t0}")
        X, y = X.to(device), y.to(device)
        print(f"1: {time.monotonic()-t0}")

        # Compute prediction error
        pred = model(X)
        print(f"2: {time.monotonic()-t0}")
        loss = loss_fn(pred, y)
        print(f"3: {time.monotonic()-t0}")

        # Backpropagation
        loss.backward()
        print(f"4: {time.monotonic()-t0}")
        optimizer.step()
        print(f"5: {time.monotonic()-t0}")
        optimizer.zero_grad()
        print(f"6: {time.monotonic()-t0}")

        if True: # batch % 20 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        print(f"7: {time.monotonic()-t0}")



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



epochs = 100
best_accuracy = None
t0 = time.monotonic()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    print("Validation: ", end="")
    accuracy = test(validation_dataloader, model, loss_fn)
    if best_accuracy is None or accuracy>best_accuracy:
        best_accuracy = accuracy
        best_model = model.state_dict()
    print("Test: ", end="")
    test(test_dataloader, model, loss_fn)
dt = time.monotonic() - t0
print(f"TIME {sys.argv[1]} {dt}")
model.load_state_dict(best_model)
print(f"Loaded model with best accuracy: {(100*best_accuracy):>0.1f}%")

fn = f"model-{sys.argv[1]}.pth"
torch.save(model.state_dict(), fn)
print(f"Saved PyTorch Model State to {fn}")

#!/usr/bin/env python3

import torch
from torch.utils.data import *
from torchvision import datasets
from torchvision.transforms import *
from torchvision.io import decode_image
import pickle

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

# Save the result
with open("data/EuroSAT-datasets.dat", "wb") as fh:
    pkg = (train_dataset, validation_dataset, test_dataset)
    pickle.dump(pkg, fh)

# EOF

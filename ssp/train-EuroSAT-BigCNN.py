#!/usr/bin/env python3

import sys
import torch
from torch import nn
from torch.utils.data import *
from torchvision import datasets
from torchvision.transforms import *
from torchvision.io import decode_image
import torch.multiprocessing
import time
import pickle

from models_EuroSAT import BigCNN


def train(dataloader, model, loss_fn, optimizer):
    total_loss = 0
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

        total_loss += loss

    return total_loss


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
    #print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct



def worker_init(gpuid_queue):
    global gpu_id, train_dataset, validation_dataset, test_dataset
    gpu_id = gpuid_queue.get()
    print(f"Init worker for GPU #{gpu_id}.")



def worker_job(batch_size):
    global gpu_id
    global train_dataset, validation_dataset, test_dataset
    global device

    print(f"Job for worker for GPU #{gpu_id}: {batch_size}")

    with open("data/EuroSAT-datasets.dat", "rb") as fh:
        (train_dataset, validation_dataset, test_dataset) = pickle.load(fh)

    # Create data loaders.
    train_dataloader      = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader       = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    torch.accelerator.set_device_index(gpu_id)

    model = BigCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    results = dict(gpuid=gpu_id, epochs=[])

    epochs = 250
    best_accuracy = None
    t0 = time.monotonic()
    for t in range(epochs):
        loss = train(train_dataloader, model, loss_fn, optimizer)
        val_acc = test(validation_dataloader, model, loss_fn)
        if best_accuracy is None or val_acc>best_accuracy:
            best_accuracy = val_acc
            best_model = model.state_dict()
        test_acc = test(test_dataloader, model, loss_fn)
        results["epochs"].append((float(loss), val_acc, test_acc))
    dt = time.monotonic() - t0
    results["time"] = dt
    results["best_accuracy"] = best_accuracy
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), f"model-{batch_size}.pth")
    with open(f"results-{batch_size}.dat", "wb") as fh:
        pickle.dump(results, fh)


def main():
    N_GPUS = torch.accelerator.device_count()

    gpuid_queue = torch.multiprocessing.SimpleQueue()
    for i in range(N_GPUS):
        gpuid_queue.put(i)

    gpu_pool = torch.multiprocessing.Pool(N_GPUS, worker_init, [gpuid_queue])

    gpu_pool.map(worker_job, [32, 64, 128, 256, 512, 1024, 2048, 4096])




if __name__=="__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()

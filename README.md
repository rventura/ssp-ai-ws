# Machine learning examples for the ISU SSP workshop on AI

## Description

This repository contains supplementary material for the author's workshop on Artificial Intelligence (AI) lectured at the Space Studies Program (SSP) of the International Space University (https://www.isunet.edu/). It features a set of Jupyter notebooks showing simple examples of neural network models and their use.

The provided examples are described below:
* `1_classifier-MINST.ipynb` -- classification of handwritten digits (MNIST dataset)
* `2_classifier-EuroSAT.ipynb` -- classification of satellite images (EuroSAT dataset)
* `3_autoencoder-MNIST.ipynb` -- unsupervised learning using an autoencoder (MNIST dataset)

All of these notebooks reside in the `ssp` folder inside the repository,

## Installation and startup instructions

There are two possible ways to install the software: __locally__, provided your computer has a modern Python installed, or through a __docker__ container, probided you have docker installed.

### Local installation

To install the required software locally, open a shell and change directory to the repository root. Then just executr the installation script by running
```
./setup_local.sh
```

To startup Jupyter, just run
```
./run_local.sh
```
This should open a browser automatically. If not, just open one of the URLs provided below the sentence "Or copy and paste one of these URLs:".

### Docker installation

To install the required software locally, open a shell and change directory to the repository root. Then just executr the installation script by running
```
./setup_docker.sh
```

To startup Jupyter, just run
```
./run_docker.sh
```
Now, open a web browser and open one of the URLs provided below the sentence "Or copy and paste one of these URLs:".

## Author

Rodrigo Ventura\
Institute for Systems and Robotics, Instituto Superior Técnico, Lisbon, Portugal\
https://wp.isr.tecnico.ulisboa.pt/rventura \
https://linkedin.com/in/rodrigo-venturas

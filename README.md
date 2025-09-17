# Investigating the Impact of Hyperparameters on Deep Neural Networks

### Overview

This repository contains my work for investigating the impact of hyperparameters on deep neural networks, as part of an assignment from NTU AI6103 Deep Learning & Applications class.

### Introduction

Deep neural networks learn by iteratively minimizing a loss function and adjusting their weights through gradient-based optimization. The magnitude of these updates is controlled by hyperparameters such as the learning rate and the learning rate schedule. However, when networks are highly overparameterized, they may overfit to the training data and fail to generalize to unseen examples. To address this issue, regularization techniques such as weight decay and batch normalization are employed, both of which are also controlled by hyperparameters.

This report investigates the impact of optimization and regularization hyperparameters on deep neural network training. The ResNet-18 architecture is used as a baseline model to systematically compare the effects of different hyperparameter choices. Experiments are conducted on the CIFAR-10 dataset.

### How to Run

### Results
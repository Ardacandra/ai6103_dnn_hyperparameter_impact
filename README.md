# Investigating the Impact of Hyperparameters on Deep Neural Networks

### Overview

This repository contains my work for investigating the impact of hyperparameters on deep neural networks, as part of an assignment from NTU AI6103 Deep Learning & Applications class.

### Abstract


### How to Run

1. Download the CIFAR-10 dataset

```
python load_cifar.py --config configs/default.yaml
```

2. Train ResNet-18 network based on the specified configuration

```
python train.py --config configs/default.yaml
```

3. Get the trained network result on the test set

```
python test.py --config configs/default.yaml
```

### Results
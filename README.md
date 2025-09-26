# Investigating the Impact of Hyperparameters on Deep Neural Networks

### Overview

This repository contains my work for investigating the impact of hyperparameters on deep neural networks, as part of an assignment from NTU AI6103 Deep Learning & Applications class.

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

### Project Summary

This project investigates how different hyperparameter choices affect the training and evaluation of a **ResNet-18** model on the **CIFAR-10** dataset. The experiments cover:  

- **Learning rates**: `0.1`, `0.01`, `0.001`  
- **Learning rate schedulers**: constant vs. cosine annealing  
- **Weight decay coefficients**: `5×10⁻⁴` and `1×10⁻²`  
- **Batch normalization variants**: PyTorch `BatchNorm2d` vs. a custom implementation without gradient flow through statistics  

### Key Results  

- **Best configuration:**  
  - Learning rate = `0.01`  
  - Scheduler = *cosine annealing*  
  - Weight decay = `1×10⁻²`  
  - Trained for `200 epochs`  

- **Performance:**  
  - **Test Accuracy:** `94.76%`  
  - **Test Loss:** `0.193`  

- Removing gradient flow from batch normalization severely degrades performance, confirming the importance of statistics in optimization.  
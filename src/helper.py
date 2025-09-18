# import all libraries
import torch

import torchvision
import torchvision.transforms as transforms

import os

def load_cifar_dataset(
    root,
    transform_train,
    transform_test,
    train_batch_size=128,
    test_batch_size=256
):
    # ensure the dataset is downloaded once
    if not os.path.exists(os.path.join(root, "cifar-10-batches-py")):
        download = True
    else:
        download = False

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=download, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=download, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader
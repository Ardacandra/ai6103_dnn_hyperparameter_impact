import torch

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import os
import argparse
import yaml
import logging
import matplotlib.pyplot as plt

from src.model import *
from src.helper import *

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    #mkdir for output if not exists
    if not os.path.exists(cfg["output_path"]):
        os.mkdir(cfg["output_path"])

    # --- Setup logging ---
    logging.basicConfig(
        filename=os.path.join(cfg["output_path"], f"{cfg['run_id']}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Main")
    logger.info(f"starting network testing...")
    logger.info(f"test parameters : {cfg}")

    if cfg["data_augmentation"]:
        # these are commonly used data augmentations
        # random cropping and random horizontal flip
        # lastly, we normalize each channel into zero mean and unit standard deviation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainloader, valloader, testloader = load_cifar_dataset(
        cfg["data_path"],
        transform_train,
        transform_test,
        subset_size=cfg["subset"],
    )

    net = ResNet18().to(cfg["device"])
    net.load_state_dict(torch.load(cfg["state_dict_path"]))
    logger.info(f"model state_dict loaded from {cfg['state_dict_path']}")

    criterion = nn.CrossEntropyLoss().to(cfg["device"])
    test_loss, test_acc = test(None, net, criterion, testloader, cfg["device"])
    logger.info(f"{cfg['run_id']} model test loss : {test_loss:.6f}, accuracy : {test_acc:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
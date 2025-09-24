import torch

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import yaml

from src.helper import *

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    #download cifar dataset
    trainloader, valloader, testloader = load_cifar_dataset(
        root = cfg["data_path"],
        transform_train=None,
        transform_test=None,
    )
    print(f"CIFAR-10 data downloaded to {cfg['data_path']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)  
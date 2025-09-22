import torch

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import os
import argparse
import yaml
import logging

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
    logging.info(f"starting network training...")
    logging.info(f"training parameters : {cfg}")

    # these are commonly used data augmentations
    # random cropping and random horizontal flip
    # lastly, we normalize each channel into zero mean and unit standard deviation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainloader, testloader = load_cifar_dataset(
        cfg["data_path"],
        transform_train,
        transform_test
    )

    net = ResNet18().to(cfg["device"])
    criterion = nn.CrossEntropyLoss().to(cfg["device"])
    optimizer = optim.SGD(net.parameters(), lr=cfg["learning_rate"],
                        momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(cfg['epoch']):
        train_loss, train_acc = train(epoch, net, criterion, trainloader, scheduler, cfg["device"], optimizer)
        test_loss, test_acc = test(epoch, net, criterion, testloader, cfg["device"])
        
        logging.info(("Epoch : %3d, training loss : %0.4f, training accuracy : %2.2f, test loss " + \
        ": %0.4f, test accuracy : %2.2f") % (epoch, train_loss, train_acc, test_loss, test_acc))  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)  
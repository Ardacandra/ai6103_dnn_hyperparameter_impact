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
    logger.info(f"starting network training...")
    logger.info(f"training parameters : {cfg}")

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
    criterion = nn.CrossEntropyLoss().to(cfg["device"])
    optimizer = optim.SGD(
        net.parameters(), 
        lr=cfg["learning_rate"],
        momentum=cfg['momentum'], 
        weight_decay=cfg['weight_decay']
    )
    if cfg["learning_rate_schedule"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        scheduler = None

    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []
    for epoch in range(1, cfg['epoch']+1):
        train_loss, train_acc = train(epoch, net, criterion, trainloader, scheduler, cfg["device"], optimizer, logger)
        val_loss, val_acc = test(epoch, net, criterion, valloader, cfg["device"])

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        
        logger.info(("Epoch : %3d, training loss : %0.4f, training accuracy : %2.2f, validation loss " + \
        ": %0.4f, validation accuracy : %2.2f") % (epoch, train_loss, train_acc, val_loss, val_acc))  

    csv_path = os.path.join(cfg["output_path"], f"{cfg['run_id']}_train_history.csv")
    df_hist = pd.DataFrame()
    df_hist['epoch'] = range(1, len(train_loss_hist)+1)
    df_hist['train_loss'] = train_loss_hist
    df_hist['train_acc'] = train_acc_hist
    df_hist['val_loss'] = val_loss_hist
    df_hist['val_acc'] = val_acc_hist
    df_hist.to_csv(csv_path, index=False)
    logger.info(f"training and validation loss and accuracy saved to {csv_path}")
    
    logger.info("saving training loss and accuracy plots...")

    #mkdir for plots
    plot_path = os.path.join(cfg["output_path"], "plots/")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    plt.plot(range(1, len(train_loss_hist)+1), train_loss_hist, 'b')
    plt.plot(range(1, len(val_loss_hist)+1), val_loss_hist, 'r')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    # plt.title("ResNet18: Loss vs Number of epochs")
    plt.legend(['train', 'validation'])
    plt.savefig(os.path.join(plot_path, f"{cfg['run_id']}_train_loss.png"), bbox_inches="tight")
    plt.close()

    plt.plot(range(1, len(train_acc_hist)+1), train_acc_hist, 'b')
    plt.plot(range(1, len(val_acc_hist)+1), val_acc_hist, 'r')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    # plt.title("ResNet18: Accuracy vs Number of epochs")
    plt.legend(['train', 'validation'])
    plt.savefig(os.path.join(plot_path, f"{cfg['run_id']}_train_acc.png"), bbox_inches="tight")
    plt.close()

    logger.info(f"plots saved to {plot_path}")

    trained_model_state_path = os.path.join(cfg["output_path"], f"{cfg['run_id']}_state_dict.pth")
    torch.save(net.state_dict(), trained_model_state_path)
    logger.info(f"trained model state_dict saved to {trained_model_state_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)  
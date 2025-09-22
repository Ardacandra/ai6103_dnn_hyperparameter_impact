# import all libraries
import torch

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset

import os

def load_cifar_dataset(
    root,
    transform_train,
    transform_test,
    train_batch_size=128,
    test_batch_size=256,
    subset_size=None
):
    # ensure the dataset is downloaded once
    if not os.path.exists(os.path.join(root, "cifar-10-batches-py")):
        download = True
    else:
        download = False

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=download, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=download, transform=transform_test
    )

    # for faster iteration during development
    if subset_size is not None:
        trainset = Subset(trainset, range(subset_size))
        testset = Subset(testset, range(subset_size))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader

# Training
def train(epoch, net, criterion, trainloader, scheduler, device, optimizer, logger):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx+1) % 50 == 0:
          logger.info("iteration : %3d, loss : %0.4f, accuracy : %2.2f" % (batch_idx+1, train_loss/(batch_idx+1), 100.*correct/total))

    if scheduler is not None:
        scheduler.step()

    return train_loss/(batch_idx+1), 100.*correct/total

def test(epoch, net, criterion, testloader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss/(batch_idx+1), 100.*correct/total

def save_checkpoint(net, acc, epoch, logger):
    # Save checkpoint.
    logger.info('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')
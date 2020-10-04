import time

import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms

import model as mdl

device = "cpu"
torch.set_num_threads(4)

batch_size = 256  # batch for one node


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    iter_number = 1
    epoch_loss = 0
    forward_time = 0
    backward_time = 0
    total_time = 0
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()
        optimizer.zero_grad()
        predictions = model(data)
        forward_time += (time.time() - start_time)

        start_time_backward = time.time()
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()
        backward_time += (time.time() - start_time_backward)
        total_time += (time.time() - start_time)

        epoch_loss += loss

        if iter_number % 20 == 0:
            epoch_loss = epoch_loss / 20
            print('Training loss after {} epochs is {}'.format(iter_number, epoch_loss))
            epoch_loss = 0
            if iter_number != 20:
                print('Forward Pass time in iter {} is {}'.format(iter_number, forward_time / 20.0))
                print('Backward Pass time in iter {} is {}'.format(iter_number, backward_time / 20.0))
                print('Average Pass time in iter {} is {}'.format(iter_number, total_time / 20.0))
            forward_time = 0
            backward_time = 0
            total_time = 0
        iter_number += 1


def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                    download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(training_set,
                                               num_workers=2,
                                               batch_size=batch_size,
                                               sampler=None,
                                               shuffle=True,
                                               pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)


if __name__ == "__main__":
    main()

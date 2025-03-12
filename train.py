import torch
from model.model import *
from git_temp.RESNET_BRAIN_TUMOR_CLASSIFICATION.train import *
from test.test import *
from git_temp.RESNET_BRAIN_TUMOR_CLASSIFICATION.plot import *
import torchvision
def train(model, epochs, optimizer, criterion, scheduler, device, trainloader, valloader):
    train_loss = []
    train_acc = []
    validation_loss = []
    validation_accuracy = []

    for i in range(epochs):
        running_loss = 0.0
        running_correct = 0
        total = 0

        model.train()
        for images, targets in trainloader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            running_correct += (pred == targets).sum().item()
            total += targets.size(0)
        scheduler.step()

        train_loss.append(running_loss / len(trainloader))
        train_acc.append(running_correct / total)

        running_val_loss = 0.0
        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for images, targets in valloader:
                images, targets = images.to(device), targets.to(device)

                output = model(images)
                pred = torch.argmax(output, dim=1)

                correct += (pred == targets).sum().item()
                running_val_loss += criterion(output, targets).item()
                total += targets.size(0)
            acc = correct / total
            validation_accuracy.append(acc)
            validation_loss.append(running_val_loss / len(valloader))
        print(f'epoch {i+1}, train loss {train_loss[-1]}, train accuracy {train_acc[-1]}, validation loss {validation_loss[-1]}, validation accuracy {validation_accuracy[-1]}')

    return train_loss, train_acc, validation_loss, validation_accuracy


    




 
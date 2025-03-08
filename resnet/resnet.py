import logging
from matplotlib.pylab import random_sample
from matplotlib.transforms import Transform
from PIL import Image
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset 
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from skimage.io import imread
import numpy as np 
import random
import pathlib
import glob
from torch.utils.data import DataLoader
from torch import optim
logging.basicConfig(level=logging.INFO)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BrainTumorDataset(Dataset):
    def __init__(self, root, transform=None):

        super().__init__()
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Get class folders and sort them for consistent label assignment
        class_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        logging.info(f"Found {len(class_dirs)} classes: {class_dirs}")

        for label, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir] = label
            class_dir_path = os.path.join(root, class_dir)
            
            # Get all valid image files in the class directory
            for img_path in glob.glob(os.path.join(class_dir_path, "*")):
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        logging.info(f"Loaded {len(self.image_paths)} images across {len(class_dirs)} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Get image and label at index"""
        img_path = self.image_paths[index]
        label = self.labels[index]

        try:
            
            image = Image.open(img_path).convert('RGB')
            
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            raise


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = nn.Sequential()
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes,out_planes,1,stride),
                nn.BatchNorm2d(out_planes)
            )
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out
        
        
class Bottleneck(nn.Module):
    expansion = 4 
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # skip connection
        out = self.relu(out)  # final activation
        
        return out
        
        
        
data_Transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456,0.456,0.406],std=[0.229, 0.224, 0.225])
    
])


        
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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


def test(model,testloader,device):
    correct = 0
    total = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for images,targets in testloader:
            images,targets = images.to(device),targets.to(device)
            output = model(images)
            preds = torch.argmax(output,dim=1)
            correct = (preds == targets).sum().item()
            total+=targets.size(0)
    print(f"The number of corrected outputs {correct},with the accuracy of {correct/total} ")
    


def plotvalue(train_loss,train_acc,val_loss,val_acc):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend()
    plt.show()
    

    
    
def testresnet():
        model =  ResNet34()
        x = torch.rand(1,3,224,224)
        y = model(x)
        print(y.shape)
        softmax = nn.Softmax(dim=1)
        value = softmax(y)
    
        print(torch.max(value,dim=1))
    
if __name__ == '__main__':
    model = ResNet(Bottleneck,[3,4,6,3]).to(device)
    print(model)


    train_dataset = BrainTumorDataset('data\MRI_classification\Training',data_Transform)
    test_dataset = BrainTumorDataset('data\MRI_classification\Testing',data_Transform)

    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=2)
    test_loader = DataLoader(test_dataset,batch_size=16,shuffle=True,num_workers=2)


    epochs= 15
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=0.0001)

    print("Starting of training ")
    train_losses, train_acc, val_losses, val_acc = train(model, epochs, optimizer, criterion, scheduler, device, train_loader, test_loader)
    print("Trainning finished")

    test(model,test_loader,device)
    
    plotvalue(train_losses,train_acc,val_losses,val_acc)


    torch.save(model.state_dict(),'ResNetT50')

    def ResNet18():
        return ResNet(BasicBlock, [2, 2, 2, 2])

    def ResNet34():
        return ResNet(BasicBlock, [3, 4, 6, 3])

    def ResNet50():
        return ResNet(Bottleneck,[3,4,6,3])#resnet 50 uses bottle nect

   
    
    #testresnet()









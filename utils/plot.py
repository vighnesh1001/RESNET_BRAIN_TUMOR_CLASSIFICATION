from datasets.datasets import *
import torchvision
import torch
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
    
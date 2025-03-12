import torchvision
from git_temp.RESNET_BRAIN_TUMOR_CLASSIFICATION.datasets import *
from model.model import *
from git_temp.RESNET_BRAIN_TUMOR_CLASSIFICATION.train import *
from test.test import *
from git_temp.RESNET_BRAIN_TUMOR_CLASSIFICATION.plot import *
import torchvision
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
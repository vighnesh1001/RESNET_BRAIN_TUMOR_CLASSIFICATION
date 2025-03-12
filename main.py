from git_temp.RESNET_BRAIN_TUMOR_CLASSIFICATION.datasets import *
from model.model import *
from git_temp.RESNET_BRAIN_TUMOR_CLASSIFICATION.train import *
from test.test import *
from git_temp.RESNET_BRAIN_TUMOR_CLASSIFICATION.plot import *
import torchvision
import torch
def main():
     
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

    
def ResNet50():
            return ResNet(Bottleneck,[3,4,6,3])#resnet 50 uses bottle nect




if __name__ == '__main__':
        main()
       
        
   
import logging
from matplotlib.pylab import random_sample
from matplotlib.transforms import Transform
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt                                                                                                 
import pandas as pd
import torch.optim as optim 
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np 
import random
import torchvision
import pathlib
import glob
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torch import optim
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
            
            image = image.open(img_path).convert('RGB')
            
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            raise

data_Transform = Transform.Compose([
    Transform.Resize((224,224)),
    Transform.RandomHorizontalFlip(),
    Transform.RandomRotation(10),
    Transform.ToTensor(),
    Transform.Normalize(mean=[0.456,0.456,0.406],std=[0.229, 0.224, 0.225])
    
])
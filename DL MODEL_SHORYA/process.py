import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Define the paths to the dataset and annotations files
train_dataset_path = r'C:\Users\hp\Downloads\ShelfImages\train'
val_dataset_path = r'C:\Users\hp\Downloads\ShelfImages\train'
test_dataset_path = r'C:\Users\hp\Downloads\ShelfImages\test'
train_annotations_file = r'C:\Users\hp\Downloads\ShelfImages\annotations.csv'
val_annotations_file = r'C:\Users\hp\Downloads\ShelfImages\annotations.csv'
test_annotations_file = r'C:\Users\hp\Downloads\ShelfImages\annotations.csv'

# Define the input size of the model
input_size = (224, 224)

# Define the number of classes
num_classes = 4

# Define the batch size
batch_size = 16

# Define the transforms for data augmentation
train_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a custom dataset
class ProductDataset(Dataset):
    def __init__(self, dataset_path, annotations_file, transforms):
        self.dataset_path = dataset_path
        self.annotations_df = pd.read_csv(annotations_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.annotations_df.iloc[idx, 0])
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        boxes = torch.tensor(self.annotations_df.iloc[idx, 1:5].values, dtype=torch.float32)
        labels = torch.tensor(self.annotations_df.iloc[idx, 5], dtype=torch.int64)
        return img, boxes, labels

# Create the datasets and dataloaders
train_dataset = ProductDataset(train_dataset_path, train_annotations_file, train_transforms)
val_dataset = ProductDataset(val_dataset_path, val_annotations_file, val_transforms)
test_dataset = ProductDataset(test_dataset_path, test_annotations_file, val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F
import numpy as np
import os
from PIL import Image
from sklearn.metrics import average_precision_score
import pandas as pd

# Set the paths to the dataset and annotations files
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

# Define the number of epochs
epochs = 50

# Define the learning rate
learning_rate = 0.001

# Define the dropout rate
dropout_rate = 0.5

# Define the pre-trained model
base_model = models.resnet50(pretrained=True)
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Linear(num_ftrs, num_classes+1)

# Define the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
base_model = base_model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)

# Define the data transformations for training and validation
train_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the data loaders for training and validation
train_data = datasets.ImageFolder(train_dataset_path, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = datasets.ImageFolder(val_dataset_path, transform=val_transforms)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Train the model
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = base_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print('Epoch {}/{} Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss))

# Evaluate the model on the validation set
base_model.eval()

y_true = []
y_score = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = base_model(inputs)

        y_true.extend(labels.cpu().numpy().tolist())
        y_score.extend(torch.softmax(outputs, dim=1)[:, 1:].cpu().numpy().tolist())

val_ap = average_precision_score(to_categorical(y_true, num_classes+1), y_score, average='macro')
print('Validation mAP: {:.4f}'.format(val_ap))

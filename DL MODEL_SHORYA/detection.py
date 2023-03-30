import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

# Define the paths to the dataset and annotations files
test_dataset_path = r'C:\Users\hp\Downloads\ShelfImages\test'
test_annotations_file = r'C:\Users\hp\Downloads\ShelfImages\annotations.csv'

# Define the input size of the model
input_size = (224, 224)

# Define the number of classes
num_classes = 4

# Load the trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('model.py'))

# Set the model to evaluation mode
model.eval()

# Define a function to read an image and its annotations
def read_image_and_annotations(image_path, annotations_path):
    image = cv2.imread(image_path)
    annotations_df = pd.read_csv(annotations_path)
    annotations = []
    for i, row in annotations_df.iterrows():
        if row['img_name'] == os.path.basename(image_path):
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            class_id = row['class_id']
            annotations.append((x1, y1, x2, y2, class_id))
    return image, np.array(annotations)

# Define a function to convert the annotations to the format expected by the model
def convert_annotations(annotations, input_size):
    boxes = []
    labels = []
    for annotation in annotations:
        x1, y1, x2, y2, class_id = annotation
        box = [x1, y1, x2, y2]
        boxes.append(box)
        labels.append(class_id)
    boxes = torch.FloatTensor(boxes)
    labels = torch.LongTensor(labels)
    return boxes, labels

# Define a function to perform inference on an image
def detect_products(image_path, annotations_path):
    # Read the image and annotations
    image, annotations = read_image_and_annotations(image_path, annotations_path)

    # Convert the annotations to the format expected by the model
    boxes, labels = convert_annotations(annotations, input_size)

    # Convert the image to a PyTorch tensor and normalize it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)

    # Perform inference
    with torch.no_grad():
        outputs = model([image])

    # Filter the detections with a confidence threshold
    detections = outputs[0]['boxes'][outputs[0]['scores'] > 0.5].cpu().numpy()
    labels = outputs[0]['labels'][outputs[0]['scores'] > 0.5].cpu().numpy()

    # Generate insights
    total_products = detections.shape[0]
    brand_products = np.sum(labels == 1)  # assume brand class ID is 1
    competing_products = total_products - brand_products
    insights = {
        'Total products': total_products,
        'Brand products': brand_products,
        'Competing products': competing_products
    }

    return insights

# Perform inference on a test image
image_path = r'C:\Users\hp\Downloads\ShelfImages\C1_P02_N1_S5_1.jpg'
annotations_path = 'annotations.csv'
insights = detect_products(image_path, annotations_path)
print(insights)

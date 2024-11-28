import os
import glob
import lzma
import pickle
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
from scripts.models2 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

# Constants
COLOR = [[255, 0, 0], [0, 0, 255], [0, 255, 255]]
IMAGE_SIZE = (224, 224)

# Class to hold image and color data
class SnapshotToTrain:
    def __init__(self, image, color_followed, current_controls):
        self.image = image
        self.color_followed = color_followed
        self.current_controls = current_controls

# Function to process snapshot pairs
def process_snapshot_pair(snapshot1, snapshot2):
    if snapshot1.image is not None and snapshot2.image is not None:
        resized_image1 = cv2.resize(snapshot1.image, IMAGE_SIZE)
        resized_image2 = cv2.resize(snapshot2.image, IMAGE_SIZE)
        resized_image1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2RGB)
        resized_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2RGB)
        concatenated_image = np.concatenate((resized_image1, resized_image2), axis=2)
        return concatenated_image, snapshot2.current_controls
    else:
        print("Warning: Missing image in one or both snapshots")

# Read and preprocess data
snapshots = []
for record in glob.glob("*.npz"):
    try:
        with lzma.open(record, "rb") as file:
            data = pickle.load(file)
            print(f"Read {len(data)} snapshots from {record}")
            npz_number = int(record.split("_")[1].split(".")[0])
            color_followed = COLOR[npz_number]
            for i in range(len(data) - 1):
                image, controls = process_snapshot_pair(data[i], data[i + 1])
                snapshots.append(SnapshotToTrain(image, color_followed, controls))
    except EOFError:
        print("Error: Compressed file ended before the end-of-stream marker was reached.")
    except Exception as e:
        print(f"An error occurred: {e}")
print(f"Total concatenated snapshots read: {len(snapshots)}")

# Data augmentation
def augment_data(snapshots):
    snapshots_augmented = []
    for snap in snapshots:
        snapshots_augmented.append(snap)
        flipped_img = snap.image[:, ::-1, :]
        flipped_ctrl = (snap.current_controls[0], snap.current_controls[1], snap.current_controls[3], snap.current_controls[2])
        snapshots_augmented.append(SnapshotToTrain(flipped_img, snap.color_followed, flipped_ctrl))
    return snapshots_augmented

snapshots_augmented = augment_data(snapshots)

# Data distribution analysis
data_controls = [snap.current_controls for snap in snapshots_augmented]
class_counts = Counter(data_controls)
print("Class Counts:", class_counts)

# Split data into training and testing sets
train_snapshots, test_snapshots = train_test_split(snapshots_augmented, test_size=0.2, random_state=42)
print(f"Training set size: {len(train_snapshots)}, Test set size: {len(test_snapshots)}")

# Custom collate function for DataLoader
def my_collate(batch):
    images = np.array([item.image for item in batch])
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    colors = torch.tensor([item.color_followed for item in batch], dtype=torch.float32)
    controls = torch.tensor([item.current_controls for item in batch], dtype=torch.float32)
    return images, colors, controls

# DataLoaders
batch_size = 16
train_loader = DataLoader(train_snapshots, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
test_loader = DataLoader(test_snapshots, batch_size=batch_size, shuffle=False, collate_fn=my_collate)

# Model, loss, optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleImageColorNet(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training and evaluation
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, colors, controls in train_loader:
        images, colors, controls = images.to(device), colors.to(device), controls.to(device)
        optimizer.zero_grad()
        outputs = model(images, colors)
        loss = criterion(outputs, controls)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, colors, controls in test_loader:
            images, colors, controls = images.to(device), colors.to(device), controls.to(device)
            outputs = model(images, colors)
            loss = criterion(outputs, controls)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += controls.size(0)
            correct += (predicted == torch.argmax(controls, dim=1)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "model.pth")

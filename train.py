import os
import glob
import lzma
import pickle
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from scripts.models2 import SimpleImageColorNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (227, 227)
COLORS = [[255, 0, 0], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], 
          [255, 0, 255], [0, 0, 0], [255, 255, 255], [128, 128, 128], [120, 120, 120], [255, 0, 0]]

class SnapshotToTrain:
    def __init__(self, image, color_followed, current_controls):
        self.image = image
        self.color_followed = color_followed
        self.current_controls = current_controls

def preprocess_image_pair(image1, image2):
    """Resize, concatenate, and normalize snapshot pair."""
    resized_image1 = cv2.resize(image1, IMAGE_SIZE)
    resized_image2 = cv2.resize(image2, IMAGE_SIZE)
    concatenated_image = np.concatenate((resized_image1, resized_image2), axis=2)  # 6 channels
    concatenated_image = concatenated_image.transpose(2, 0, 1) / 255.0  # Normalize and reorder to (C, H, W)
    return torch.tensor(concatenated_image, dtype=torch.float32)

# Load snapshots
snapshots = []
for record in glob.glob("*.npz"):
    try:
        with lzma.open(record, "rb") as file:
            data = pickle.load(file)
            npz_number = int(record.split("_")[1].split(".")[0])
            color_followed = COLORS[npz_number]

            for i in range(len(data) - 1):
                image = preprocess_image_pair(data[i].image, data[i + 1].image)
                controls = data[i + 1].current_controls
                snapshots.append(SnapshotToTrain(image, color_followed, controls))
    except Exception as e:
        print(f"Error loading {record}: {e}")

print(f"Loaded {len(snapshots)} snapshot pairs.")

# Data augmentation
def augment_data(snapshots):
    augmented = []
    for snap in snapshots:
        augmented.append(snap)  # Original
        flipped_img = snap.image.flip(dims=[2])  # Horizontal flip
        flipped_ctrl = (snap.current_controls[0], snap.current_controls[1], 
                        snap.current_controls[3], snap.current_controls[2])  # Swap left-right
        augmented.append(SnapshotToTrain(flipped_img, snap.color_followed, flipped_ctrl))
    return augmented

snapshots = augment_data(snapshots)

# Split into train and test sets
train_snapshots, test_snapshots = train_test_split(snapshots, test_size=0.2, random_state=42)

# Custom collate function
def my_collate(batch):
    images = torch.stack([item.image for item in batch])
    colors = torch.tensor([item.color_followed for item in batch], dtype=torch.float32)
    controls = torch.tensor([item.current_controls for item in batch], dtype=torch.long)  # For CrossEntropyLoss
    return images, colors, controls

# DataLoader
batch_size = 32
train_loader = DataLoader(train_snapshots, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
test_loader = DataLoader(test_snapshots, batch_size=batch_size, shuffle=False, collate_fn=my_collate)

# Model, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleImageColorNet(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

# Training loop
best_model_state = None
best_test_loss = float('inf')

for epoch in range(5):
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
            test_loss += criterion(outputs, controls).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == controls).sum().item()
            total += controls.size(0)
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = model.state_dict()

    scheduler.step(test_loss)
    print(f"Epoch [{epoch+1}/100]: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Accuracy={accuracy:.2f}%")

# Save best model
if best_model_state:
    torch.save(best_model_state, "model.pth")

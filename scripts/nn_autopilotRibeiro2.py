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
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

# Color mappings
COLOR = [[255, 0, 0], [0, 0, 255], [0, 255, 255], [255, 0, 0], [0, 0, 255]]

# Free GPU memory
torch.cuda.empty_cache()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IMAGE_SIZE = (227, 227)  # Resize images
BATCH_SIZE = 64
NUM_EPOCHS = 500
LEARNING_RATE = 0.001

# Data container class
class SnapshotToTrain:
    def __init__(self, image, color_followed, current_controls):
        self.image = image
        self.color_followed = color_followed
        self.current_controls = current_controls

# Data preprocessing
snapshots = []
for record in glob.glob("*.npz"):
    try:
        with lzma.open(record, "rb") as file:
            data = pickle.load(file)
            print(f"Read {len(data)} snapshots from {record}")
            
            npz_number = int(record.split("_")[1].split(".")[0])
            color_followed = COLOR[npz_number]

            for i in range(len(data) - 1):
                img1, img2 = data[i].image, data[i + 1].image
                controls = data[i + 1].current_controls

                if img1 is not None and img2 is not None:
                    img1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), IMAGE_SIZE) / 255.0
                    img2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), IMAGE_SIZE) / 255.0
                    concatenated_img = np.concatenate((img1, img2), axis=2)
                    snapshots.append(SnapshotToTrain(concatenated_img, color_followed, controls))
    except Exception as e:
        print(f"Error processing {record}: {e}")

# Augment data
def augment_data(snapshots):
    augmented = []
    for snap in snapshots:
        augmented.append(snap)
        flipped_img = snap.image[:, ::-1, :]
        flipped_ctrl = (snap.current_controls[0], snap.current_controls[1], snap.current_controls[3], snap.current_controls[2])
        augmented.append(SnapshotToTrain(flipped_img, snap.color_followed, flipped_ctrl))
    return augmented

snapshots = augment_data(snapshots)

# Split data
train_snapshots, test_snapshots = train_test_split(snapshots, test_size=0.2, random_state=42)

# DataLoader
def my_collate(batch):
    images = np.array([item.image for item in batch]) / 255.0
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    colors = torch.tensor([item.color_followed for item in batch], dtype=torch.float32)
    controls = torch.tensor([item.current_controls for item in batch], dtype=torch.float32)
    return images, colors, controls

train_loader = DataLoader(train_snapshots, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
test_loader = DataLoader(test_snapshots, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)

model = SimpleImageColorNet().to(device)

# Loss and optimizer
class_counts = Counter([tuple(snap.current_controls) for snap in train_snapshots])
weights = torch.tensor([1 / max(class_counts.values()) * class_counts[tuple(c)] for c in range(4)], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
train_losses, test_losses, accuracies = [], [], []
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for images, colors, controls in train_loader:
        images, colors, controls = images.to(device), colors.to(device), controls.to(device)
        optimizer.zero_grad()
        outputs = model(images, colors)
        loss = criterion(outputs, controls)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Evaluate
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, colors, controls in test_loader:
            images, colors, controls = images.to(device), colors.to(device), controls.to(device)
            outputs = model(images, colors)
            test_loss += criterion(outputs, controls).item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == controls).all(dim=1).sum().item()
            total += controls.size(0)
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    accuracy = correct / total
    accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), "improved_model.pth")
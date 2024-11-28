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
from sklearn.model_selection import train_test_split
from collections import Counter
from scripts.models2 import *
import matplotlib.pyplot as plt

# Free CUDA memory
torch.cuda.empty_cache()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IMAGE_SIZE = (227, 227)
BATCH_SIZE = 64
NUM_EPOCHS = 500
LEARNING_RATE = 0.001
COLOR = [[255, 0, 0], [0, 0, 255], [0, 255, 255], [255, 0, 0], [0, 0, 255]]


# Class to store snapshot data
class SnapshotToTrain:
    def __init__(self, image, color_followed, current_controls):
        self.image = image
        self.color_followed = color_followed
        self.current_controls = current_controls


# Function to process each pair of snapshots
def process_snapshot_pair(snapshot1, snapshot2):
    if snapshot1.image is not None and snapshot2.image is not None:
        # Resize images
        resized_image1 = cv2.resize(snapshot1.image, IMAGE_SIZE)
        resized_image2 = cv2.resize(snapshot2.image, IMAGE_SIZE)

        # Convert BGR to RGB
        resized_image1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2RGB)
        resized_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2RGB)

        # Concatenate along the color channel
        concatenated_image = np.concatenate((resized_image1, resized_image2), axis=2)
        return concatenated_image, snapshot2.current_controls
    else:
        print("Warning: Missing image in one or both snapshots")
        return None, None


# Load and preprocess data
snapshots = []
for record in glob.glob("*.npz"):
    try:
        with lzma.open(record, "rb") as file:
            data = pickle.load(file)
            print(f"Read {len(data)} snapshots from {record}")

            # Determine color followed
            npz_number = int(record.split("_")[1].split(".")[0])
            color_followed = COLOR[npz_number]

            # Process snapshot pairs
            for i in range(len(data) - 1):
                image, controls = process_snapshot_pair(data[i], data[i + 1])
                if image is not None:
                    snapshots.append(SnapshotToTrain(image, color_followed, controls))

    except EOFError:
        print("Error: Compressed file ended before the end-of-stream marker was reached.")
    except Exception as e:
        print(f"An error occurred: {e}")

print(f"Total concatenated snapshots read: {len(snapshots)}")


# Augment data
def augment_data(snapshots):
    augmented_snapshots = []

    for snap in snapshots:
        image = snap.image
        color = snap.color_followed
        controls = snap.current_controls

        # Original
        augmented_snapshots.append(SnapshotToTrain(image, color, controls))

        # Flipped horizontally
        flipped_image = image[:, ::-1, :]
        flipped_controls = (controls[0], controls[1], controls[2], controls[3])
        augmented_snapshots.append(SnapshotToTrain(flipped_image, color, flipped_controls))

    return augmented_snapshots


snapshots_augmented = augment_data(snapshots)
print(f"Total augmented snapshots: {len(snapshots_augmented)}")

# Class balance
data_controls = [tuple(snap.current_controls) for snap in snapshots_augmented]
class_counts = Counter(data_controls)
print("Class Counts:", class_counts)

# Train/test split
train_snapshots, test_snapshots = train_test_split(snapshots_augmented, test_size=0.2, random_state=42)
print(f"Training snapshots: {len(train_snapshots)}, Testing snapshots: {len(test_snapshots)}")

# Calculate class weights
total_controls = sum(class_counts.values())
normalized_weights = [
    total_controls / (len(class_counts) * class_counts[cls]) for cls in class_counts
]
class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(device)

# Custom DataLoader collate function
def my_collate(batch):
    # Convert list of images to a single NumPy array
    images = np.stack([item.image for item in batch], axis=0)  # Stack along batch dimension
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to PyTorch tensor, HWC -> CHW

    # Convert colors and controls
    colors = torch.tensor([item.color_followed for item in batch], dtype=torch.float32)
    controls = torch.tensor([item.current_controls for item in batch], dtype=torch.float32)

    return images, colors, controls



# Create DataLoaders
train_loader = DataLoader(train_snapshots, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
test_loader = DataLoader(test_snapshots, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)

# Model
model = ModifiedAlexNet().to(device)
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Loss function
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)

# Optimizer and scheduler
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)

# Training loop
best_accuracy = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for images, colors, controls in train_loader:
        images, colors, controls = images.to(device), colors.to(device), controls.to(device)

        optimizer.zero_grad()
        outputs = model(images, colors).view(-1, controls.size(1))
        train_loss_batch = criterion(outputs, controls)
        train_loss_batch.backward()
        optimizer.step()

        train_loss += train_loss_batch.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, colors, controls in test_loader:
            images, colors, controls = images.to(device), colors.to(device), controls.to(device)

            outputs = model(images, colors)
            loss = criterion(outputs, controls)
            test_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).int()
            correct += (predicted == controls.int()).all(dim=1).sum().item()
            total += controls.size(0)

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    scheduler.step(test_loss)

torch.save(model.state_dict(), "model.pth")

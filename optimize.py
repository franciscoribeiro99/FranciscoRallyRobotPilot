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
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
from collections import Counter
from scripts.models2 import *
import matplotlib.pyplot as plt
import optuna

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IMAGE_SIZE = (227, 227)
NUM_EPOCHS = 500
COLOR = [[255, 0, 0], [0, 0, 255], [0, 255, 255], [255, 0, 0], [0, 0, 255],[255,0,0],[0,255,255],[0,0,255]]


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
        print("Error: Compressed file ended prematurely.")
    except Exception as e:
        print(f"An error occurred: {e}")

print(f"Total concatenated snapshots read: {len(snapshots)}")


def augment_data(snapshots):
    augmented_snapshots = []
    for snap in snapshots:
        image = snap.image
        color = snap.color_followed
        controls = np.array(snap.current_controls, dtype=np.float32)  # Convert to numpy array

        # Original
        augmented_snapshots.append(SnapshotToTrain(image, color, controls))

        # Flipped horizontally
        flipped_image = cv2.flip(image, 1)

        flipped_controls = np.array([controls[0], controls[1], controls[3], controls[2]], dtype=np.float32)  # Swap left/right
        augmented_snapshots.append(SnapshotToTrain(flipped_image, color, flipped_controls))

        # Brightened
        bright_image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        augmented_snapshots.append(SnapshotToTrain(bright_image, color, controls))

    return augmented_snapshots

#take a random snapshot and display it
random_snapshot = snapshots[np.random.randint(0, len(snapshots))]
print("Color followed:", random_snapshot.color_followed, "Controls:", random_snapshot.current_controls)

snapshots_augmented = augment_data(snapshots)
print(f"Total augmented snapshots: {len(snapshots_augmented)}")

# Split data
snapshots_train, snapshots_val = train_test_split(snapshots_augmented, test_size=0.2, random_state=42)

# Compute class weights
train_labels = [np.array(snap.current_controls).argmax() for snap in snapshots_train]
val_labels = [np.array(snap.current_controls).argmax() for snap in snapshots_val]
print("Train class distribution:", Counter(train_labels))
print("Validation class distribution:", Counter(val_labels))
class_weights = torch.tensor([1.0 / count for count in Counter(train_labels).values()]).to(device)

# Custom dataloader
def my_collate(batch):
    images = np.array([item.image for item in batch])
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
    
    colors = np.stack([item.color_followed for item in batch])  # Stack into a single numpy array
    colors = torch.tensor(colors, dtype=torch.float32)
    
    
    controls = np.stack([item.current_controls for item in batch])  # Stack controls
    controls = torch.tensor(controls, dtype=torch.float32)  # Convert to class indices
    return images, colors, controls


# Model
model = ModifiedAlexNet().to(device)
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Optuna trial for hyperparameter tuning
def train_and_evaluate_model(trial):
    # Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2)
    scheduler_name = trial.suggest_categorical("scheduler", ["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"])
    step_size = trial.suggest_int("step_size", 5, 20) if scheduler_name == "StepLR" else None

    # Dataloaders
    train_loader = DataLoader(snapshots_train, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=4)
    val_loader = DataLoader(snapshots_val, batch_size=batch_size, shuffle=False, collate_fn=my_collate, num_workers=4)

    # Model
    model = ModifiedAlexNet(dropout_rate=dropout_rate).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) if optimizer_name == "Adam" else SGD(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    # Scheduler
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Loss function
    criterion = nn.BCEWithLogitsLoss(weight=class_weights)

    # Training and validation loops
    best_accuracy = 0.0
    for epoch in range(30):  # Fewer epochs for Optuna tuning
        model.train()
        train_loss = 0.0
        for images, colors, controls in train_loader:
            images, colors, controls = images.to(device), colors.to(device), controls.to(device)
            optimizer.zero_grad()
            outputs = model(images, colors)
            loss = criterion(outputs, controls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for images, colors, controls in val_loader:
                images, colors, controls = images.to(device), colors.to(device), controls.to(device)
                outputs = model(images, colors)
                val_loss += criterion(outputs, controls).item()

                # a prediction is correct only if all controls are correct
                predicted = (torch.sigmoid(outputs) > 0.5).float()

                correct += (predicted == controls).all(dim=1).sum().item() 

                total += controls.size(0)

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total

        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        best_accuracy = max(best_accuracy, accuracy)
        print(f"Epoch [{epoch+1}/30], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return best_accuracy

# Define Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(train_and_evaluate_model, n_trials=50)

# Save results
print("Best hyperparameters:", study.best_params)
study.trials_dataframe().to_csv("optuna_results.csv")

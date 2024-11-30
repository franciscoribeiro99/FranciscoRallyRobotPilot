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


# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IMAGE_SIZE = (227, 227)
BATCH_SIZE = 64
DROP_OUT = 0.5
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
#COLOR = [[255, 0, 0],[0,255,255],[0,0,255]]
COLOR = [[255, 0, 0], [0, 0, 255], [0, 255, 255], [255, 0, 0], [0, 0, 255],[255,0,0],[0,255,255],[0,0,255],[255, 0, 0],[0,255,255],[0,0,255]]




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
            print(f"colors followed is {color_followed}")
            
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

        # adding noise on a image that is 6x227x227
        noise = np.random.normal(0, 0.1, image.shape)
        noisy_image = image + noise
        augmented_snapshots.append(SnapshotToTrain(noisy_image, color, controls))

        # augmenting the color
        color = np.array(color)
        color = color + np.random.normal(0, 0.1, color.shape)
        augmented_snapshots.append(SnapshotToTrain(image, color, controls))


        # Rotated 90 degrees
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_controls = np.array([controls[1], controls[0], controls[2], controls[3]], dtype=np.float32)  # Rotate left/right
        augmented_snapshots.append(SnapshotToTrain(rotated_image, color, rotated_controls))

        # Gaussian blur
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        augmented_snapshots.append(SnapshotToTrain(blurred_image, color, controls))

        # Contrast adjustment
        contrast_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        augmented_snapshots.append(SnapshotToTrain(contrast_image, color, controls))

        translated_image = cv2.warpAffine(image, np.float32([[1, 0, 10], [0, 1, 10]]), (image.shape[1], image.shape[0]))
        augmented_snapshots.append(SnapshotToTrain(translated_image, color, controls))
                

    return augmented_snapshots



snapshots_augmented = augment_data(snapshots)
print(f"Total augmented snapshots: {len(snapshots_augmented)}")



# Split data
snapshots_train, snapshots_val = train_test_split(snapshots_augmented, test_size=0.2, random_state=42)

# class weights
train_labels = [np.array(snap.current_controls).argmax() for snap in snapshots_train]
val_labels = [np.array(snap.current_controls).argmax() for snap in snapshots_val]
print("Train class distribution:", Counter(train_labels))

class_weights = 1.0 / np.array(list(Counter(train_labels).values()))

# Custom dataloader
def my_collate(batch):
    # Convert images to tensor
    images = np.array([item.image for item in batch])
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
    """
    # Add Gaussian noise
    noise = torch.randn(images.size()) * 0.0
    images = images + noise
    """
    
    # Convert colors to tensor
    colors = np.stack([item.color_followed for item in batch])  # Stack into a single numpy array
    colors = torch.tensor(colors, dtype=torch.float32)
    
    # Convert controls to tensor
    controls = np.stack([item.current_controls for item in batch])  # Stack controls
    controls = torch.tensor(controls, dtype=torch.float32)  # Convert to class indices

    


    return images, colors, controls

# Weighted sampler for imbalanced classes
class_sample_weights = 1.0 / np.array([26231, 4225, 3858, 3609])  # Adjust based on distribution
train_sampler = torch.utils.data.WeightedRandomSampler(
    weights=class_sample_weights, num_samples=len(snapshots_train), replacement=True
)

# Create dataloaders for training and validation
train_loader = DataLoader(snapshots_train, BATCH_SIZE, shuffle=True, collate_fn=my_collate)
val_loader = DataLoader(snapshots_val, BATCH_SIZE,shuffle=False, collate_fn=my_collate)

# Model 
model = ModifiedAlexNet(DROP_OUT).to(device)
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# Optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001, amsgrad=True)

# Loss function
criterion = nn.BCEWithLogitsLoss(weight=class_weights)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

# store all data over the epoch
train_loss_d = []
val_loss_d = []
accuracy_d = []

# Training loop
best_accuracy = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for images, colors, controls in train_loader:
        images, colors, controls = images.to(device), colors.to(device), controls.to(device)# Move to device
        optimizer.zero_grad()
        outputs = model(images, colors)
        loss = criterion(outputs, controls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_loss_d.append(train_loss) 

    # Validation
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for images, colors, controls in val_loader:
            images, colors, controls = images.to(device), colors.to(device), controls.to(device)
            outputs = model(images, colors)
            val_loss += criterion(outputs, controls).item()

            #set threshold to check
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            correct += (predicted == controls).all(dim=1).sum().item() 

            total += controls.size(0)

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    accuracy_d.append(accuracy)
    val_loss_d.append(val_loss)

    scheduler.step(val_loss)


    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_final_model.pth")

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f},Val loss: {val_loss:.4f} Accuracy: {accuracy:.2f}%")

   


# save plots with information about the model
plt.plot(train_loss_d, label="Train Loss")
plt.plot(val_loss_d, label="Validation Loss")
#plt.plot(accuracy_d, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("metrics10.png")
plt.close()

print(f"Best Validation Accuracy: {best_accuracy:.2f}%")


torch.save(model.state_dict(), "model.pth")

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
NUM_EPOCHS = 500
LEARNING_RATE = 0.001 #try 0.0001
#COLOR = [[255, 0, 0],[0,255,255],[0,0,255]]
COLOR = [
    [255, 0, 0], [0, 0, 255], [0, 255, 255], [255, 0, 0], 
    [0, 0, 255], [255, 0, 0], [0, 255, 255], [0, 0, 255], 
    [255, 0, 0], [0, 255, 255], [0, 0, 255],[255,255,0],
    [255,0,255],[0,255,0],[255,255,0],[255,0,0],[255,0,255]
]




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

            countBack=0
            allssnaps=0

            # Process snapshot pairs
            for i in range(len(data) - 1):
                image, controls = process_snapshot_pair(data[i], data[i + 1])
                if image is not None:
                    if data[i+1].current_controls[1]==1:
                        countBack+=1
                    allssnaps+=1
                    snapshots.append(SnapshotToTrain(image, color_followed, controls))

            print(f"Backward percentage is {(countBack*100)/allssnaps}")
            

    except EOFError:
        print("Error: Compressed file ended before the end-of-stream marker was reached.")
    except Exception as e:
        print(f"An error occurred: {e}")

print(f"Total concatenated snapshots read: {len(snapshots)}")



### REMOVE SOME HERE  GAUSSIAN BLUR and BRIGHTNESS

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
  

        # Gaussian blur
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        augmented_snapshots.append(SnapshotToTrain(blurred_image, color, controls))

        # Brightness
       # Brightness adjustment
        brightness = 0.5
        brightened_image = np.clip(image * brightness, 0, 255).astype(np.uint8)

        # Add to augmented snapshots
        augmented_snapshots.append(SnapshotToTrain(brightened_image, color, controls))



    return augmented_snapshots



snapshots_augmented = augment_data(snapshots)
print(f"Total augmented snapshots: {len(snapshots_augmented)}")



# Split data
snapshots_train, snapshots_val = train_test_split(snapshots_augmented, test_size=0.2, random_state=42)


# Calculate the frequency of each class
train_labels = [np.array(snap.current_controls).argmax() for snap in snapshots_train]
class_counts = Counter(train_labels)

# Total number of samples
total_samples = len(train_labels)

# Calculate weights
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

# Normalize weights (optional, to avoid very large values)
max_weight = max(class_weights.values())
class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

print("Class Weights:", class_weights)

# Convert to a tensor for PyTorch loss function
class_weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_counts))]).to(device)


# class weights
train_labels = [np.array(snap.current_controls).argmax() for snap in snapshots_train]
val_labels = [np.array(snap.current_controls).argmax() for snap in snapshots_val]
print("Train class distribution:", Counter(train_labels))



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


# Create dataloaders for training and validation
train_loader = DataLoader(snapshots_train, BATCH_SIZE, shuffle=True, collate_fn=my_collate)
val_loader = DataLoader(snapshots_val, BATCH_SIZE,shuffle=False, collate_fn=my_collate)

#augmenting data using transformer
transformer = transforms.Compose([
    #print just
    #blur  
    transforms.GaussianBlur(kernel_size=5),
    #brightness
    transforms.ColorJitter(brightness=0.5),
    #other transformations
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(5),
    #transforms.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(0.9, 1.1)),#maybe check with this one
    transforms.RandomResizedCrop(227, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#check thsi one too
])





# Model 
model = ModifiedAlexNet(DROP_OUT).to(device)
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# Optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001, amsgrad=True)

# Loss function
criterion = nn.BCEWithLogitsLoss(weight=class_weights_tensor)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

# store all data over the epoch
train_loss_d = []
val_loss_d = []
accuracy_d = []
accuracy2_d = []
train_accuracy = []

# Training loop
best_accuracy = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct, train_total = 0, 0
    for images, colors, controls in train_loader:
        images, colors, controls = images.to(device), colors.to(device), controls.to(device)# Move to device
        optimizer.zero_grad()
        outputs = model(images, colors)
        loss = criterion(outputs, controls)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (predicted == controls).all(dim=1).sum().item()
        train_total += controls.size(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        train_accuracy = 100 * train_correct / train_total
    train_loss /= len(train_loader)
    train_loss_d.append(train_loss)

    # Validation
    model.eval()
    correct, total,correct2,total2 = 0, 0,0,0
    val_loss = 0.0
    with torch.no_grad():
        for images, colors, controls in val_loader:
            images, colors, controls = images.to(device), colors.to(device), controls.to(device)
            outputs = model(images, colors)
            val_loss += criterion(outputs, controls).item()

            #set threshold to check
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            # Convert tensors to lists for better readability
            predicted_values = predicted.detach().cpu().numpy()
            real_values = controls.detach().cpu().numpy()
            # Print each prediction and the corresponding real value
             #for pred, real in zip(predicted_values, real_values):
                #print(f"Predicted: {pred.tolist()}, Real: {real.tolist()}")

            correct += (predicted == controls).all(dim=1).sum().item() 
            #check by class
            correct2+= (predicted == controls).sum().item()

            total += controls.size(0)
            total2 += controls.numel() 

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    accuracy2 = 100 * correct2 / total2
    accuracy_d.append(accuracy)
    val_loss_d.append(val_loss)
    accuracy2_d.append(accuracy2)
    scheduler.step(val_loss)
 

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f},Val loss: {val_loss:.4f} Accuracy: {accuracy:.2f}%, accuracy2: {accuracy2:.2f}%")

   


# save plots with information about the model
plt.plot(train_loss_d, label="Train Loss")
plt.plot(val_loss_d, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
plt.close()

plt.plot(accuracy_d, label="Validation Accuracy")
plt.plot(accuracy2_d, label="Validation Accuracy by class")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.close()


print(f"Best Validation Accuracy: {best_accuracy:.2f}%")


torch.save(model.state_dict(), "model.pth")

import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from data_collector import DataCollectionUI
from models2 import SimpleImageColorNet
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

import lzma, pickle
from math import sqrt
from functools import reduce

import json
import requests

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the pre-trained model
IMAGE_SIZE = (227, 227)
model_path = "model.pth"
color_to_follow=[255,0,0]

# Load model
model = SimpleImageColorNet()
try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Control action labels
output_feature_labels = ['forward', 'back', 'left', 'right']

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
])

def preprocess_input(image, color):
    """Resize, normalize, and prepare the image and color input."""
    if image.shape[-1] != 6:
        raise ValueError(f"Expected image with 6 channels, got {image.shape[-1]}.")
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).div(255.0)
    color_tensor = torch.tensor(color, dtype=torch.float32).unsqueeze(0)
    return image_tensor.unsqueeze(0).to(device), color_tensor.to(device)

class CNNMsgProcessor:
    def __init__(self, *args, **kwargs):
        self.model = model
        self.last_message = None
        self.is_simulation_finished = False

    def cnn_infer(self, message):
        if message.image is None:
            print("Error: Missing image in the message.")
            return []

        try:
            resized_image = cv2.resize(message.image, IMAGE_SIZE)
            resized_last = cv2.resize(self.last_message.image if self.last_message else resized_image, IMAGE_SIZE)
        
        
            concatenated = np.concatenate((resized_image, resized_last), axis=2)
            thresholds = [0.5, 0.5, 0.5, 0.5] 

            concatenated_tensor, color_tensor = preprocess_input(concatenated, color_to_follow)

            # Perform inference
            with torch.no_grad():
                outputs = self.model(concatenated_tensor, color_tensor)  # Raw logits
                sigmoid_outputs = torch.sigmoid(outputs)  # Apply sigmoid to outputs
                #aply threshold 0 ,1 
                predicted = (sigmoid_outputs > torch.tensor(thresholds).to(device)).float()
                predicted = predicted.detach().cpu().numpy()
                predicted.tolist()
                

            
            print(f"Resized image shape: {resized_image.shape}")
            print(f"Concatenated tensor shape: {concatenated_tensor.shape}")
            print(f"Color tensor shape: {color_tensor.shape}")
            print(f"Sigmoid outputs: {sigmoid_outputs}")

            # Return all active actions and put False the one that are not 
            active_actions = [
                (output_feature_labels[i], int(predicted[0][i])) for i in range(len(output_feature_labels))
            ]

            return active_actions

        except Exception as e:
                print(f"Inference error: {e}")
                return []
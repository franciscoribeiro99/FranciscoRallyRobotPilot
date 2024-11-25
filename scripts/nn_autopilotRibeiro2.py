import sys
import torch
import torch.nn as nn
from PyQt6 import QtWidgets
import cv2
import numpy as np
from torchvision import transforms
from data_collector import DataCollectionUI  
from models2 import SimpleImageColorNet 
import matplotlib.pyplot as plt
import time

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the pre-trained model
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
output_feature_labels = ['forward', 'back', 'right', 'left']

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
    def __init__(self):
        self.model = model
        self.last_message = None

    def cnn_infer(self, message):
        if message.image is None:
            print("Error: Missing image in the message.")
            return []

        try:
            resized_image = cv2.resize(message.image, IMAGE_SIZE)
            resized_last = cv2.resize(self.last_message.image if self.last_message else resized_image, IMAGE_SIZE)
            concatenated = np.concatenate((resized_image, resized_last), axis=2)
            concatenated_tensor, color_tensor = preprocess_input(concatenated, color_to_follow)

            with torch.no_grad():
                output = self.model(concatenated_tensor, color_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_action = torch.argmax(probabilities, dim=1).item()
            
            # print the probabilities
            print(probabilities)
            print(f"Predicted action: {output_feature_labels[predicted_action]}")
            
            return [(output_feature_labels[predicted_action], True)]
        except Exception as e:
            print(f"Inference error: {e}")
            return []


    def handle_message(self, message, data_collector):
        try:
            print("Handling message...")

            # Infer commands from the CNN model
            commands = self.cnn_infer(message)

            # Update last_message after successful processing
            self.last_message = message if message.image is not None else self.last_message

            # Send commands to the data collector
            for command, start in commands:
                data_collector.onCarControlled(command, start)

        except Exception as e:
            print(f"Error while handling message: {e}")


if __name__ == "__main__":
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)

    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)
    cnn_brain = CNNMsgProcessor()
    data_window = DataCollectionUI(cnn_brain.handle_message)
    data_window.show()
    app.exec()

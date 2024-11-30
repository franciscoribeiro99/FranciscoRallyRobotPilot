import sys
import torch
import torch.nn as nn
from PyQt6 import QtWidgets
import cv2
import numpy as np
from torchvision import transforms
from data_collector import DataCollectionUI  
from models2 import ModifiedAlexNet 
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid
import time


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Path to the pre-trained model
model_path = "model.pth"
color_to_follow=[255,0,0]

IMAGE_SIZE = (227, 227)
THRESHOLD = 0.5 

# Load model
model = ModifiedAlexNet()
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

            thresholds = [0.6, 0.6, 0.5, 0.4]  # Example thresholds for [Forward, Backward, Left, Right]
            # Perform inference
            with torch.no_grad():
                outputs = self.model(concatenated_tensor, color_tensor)  # Raw logits
                predicted = (torch.sigmoid(outputs) > torch.tensor(thresholds, device='cuda')).float()

            # Debugging output
            print(f"Outputs (logits): {outputs}")
            print(f"Predicted (processed): {predicted}")

                # Return all active actions
            active_actions = [
                (output_feature_labels[i], True) for i, p in enumerate(predicted[0]) if p > 0
            ]

            return active_actions
        except Exception as e:
            print(f"Error during inference: {e}")
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
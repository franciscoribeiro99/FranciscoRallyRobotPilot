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
import torch.nn.functional as F

import lzma, pickle
from math import sqrt
from functools import reduce

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

def compute_metrics(record_filename):
    # load record file
    with lzma.open(record_filename, "r") as f:
        sensing_messages = pickle.load(f)

    finite_differences = lambda p1, p2: sqrt((p2[0] - p1[0])**2 + (p2[2] - p1[2])**2)
    finite_differences_3 = lambda p1, p2, p3: (
        sqrt(
            ((p3[0] - p2[0]) - (p2[0] - p1[0]))**2 +
            ((p3[2] - p2[2]) - (p2[2] - p1[2]))**2
        )
    )

    car_positions = [s.car_position for s in sensing_messages]

    # compute mean speed
    car_speeds = [finite_differences(p1, p2) for p1, p2 in zip(car_positions[:-1], car_positions[1:])]
    mean_speed = sum(car_speeds) / len(car_speeds)

    # compute mean acceleration
    car_accelerations = [finite_differences_3(p1, p2, p3) for p1, p2, p3 in zip(car_positions[:-2], car_positions[1:-1], car_positions[2:])]
    mean_acceleration = sum(car_accelerations) / len(car_accelerations)

    print("Metrics computed on CNN-infered car controls:")
    print(f"Mean speed:\t{mean_speed} [units/frame]")
    print(f"Mean acceleration:\t{mean_acceleration} [units/frame²]")


class CNNMsgProcessor:
    def __init__(self, *args, **kwargs):
        self.model = model
        self.last_message = None
        self.simulation_time = int(args[0][1]) # in seconds
        self.start_time = None
        self.is_simulation_finished = False

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
        if self.start_time is None:
            # autopilot is turned on,
            # start measuring time
            self.start_time = time.time()
        else:
            # check if timeout shall
            # be issued (in seconds)
            execution_time = time.time() - self.start_time
            if execution_time >= self.simulation_time:
                print("Saving data..")
                if data_collector.saving_worker is None:
                    data_collector.saveRecord()

                    data_collector.recording = False
                    data_collector.message_processing_callback = lambda x, y: None
            else:
                try:
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
    cnn_brain = CNNMsgProcessor(sys.argv)
    data_window = DataCollectionUI(cnn_brain.handle_message, record=True, onDataSaved=compute_metrics)
    data_window.show()
    app.exec()

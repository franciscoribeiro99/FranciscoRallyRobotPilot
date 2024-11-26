import sys
import torch
import torch.nn as nn
from PyQt6 import QtWidgets
import cv2
import numpy as np
from torchvision import transforms
from data_collector import DataCollectionUI
from models import SimpleImageColorNet
import matplotlib.pyplot as plt
import time
import numpy as np

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

try:
    cnn_model = CNN(num_classes=4)
    model = ConcatModel(cnn_model, cnn_color).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded and set to evaluation mode.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
])

# Command labels corresponding to model output indices
output_feature_labels = ['forward', 'backward', 'right', 'left']

def preprocess_image(image):
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input.")
    print("Preprocessing image...")
    image = cv2.resize(image, (640, 512))  # Resize to model input dimensions
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to [channels, height, width]
    image = image.unsqueeze(0)  # Add batch dimension
    print(f"Preprocessed image shape: {image.shape}")
    return image.to(device)

def preprocess_color(color_rgb):
    if color_rgb is None or len(color_rgb) != 3:
        raise ValueError("Invalid color input.")
    print("Preprocessing color...")
    color_rgb = [c / 255.0 for c in color_rgb]  # Normalize RGB values
    color = torch.tensor(color_rgb, dtype=torch.float32).unsqueeze(0)

    return color.to(device)

    return image_tensor.unsqueeze(0).to(device), color_tensor.to(device)


class CNNMsgProcessor:
    def __init__(self):
        self.model = model
        self.last_message = None  # To store the previous valid message

    def nn_infer(self, message):
        print("Running inference...")
        try:
            image_input = preprocess_image(message.image)
            color_input = preprocess_color(color_to_follow)
        except ValueError as e:
            print(f"Preprocessing error: {e}")
            return None

        with torch.no_grad():
            outputs = self.model(image_input, color_input)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()

        print(f"Model output probabilities: {probabilities}")

        # Adjust confidence threshold dynamically (e.g., based on average)
        threshold = 0.5
        max_prob = max(probabilities)
        if max_prob > 0.8:  # If a clear decision can be made
            threshold = 0.6
        elif max_prob < 0.2:  # Avoid noise triggering commands
            print("No confident command found. Ignoring this frame.")
            return None

        # Collect commands with probabilities above the threshold
        commands = []
        for i, prob in enumerate(probabilities):
            if prob >= threshold:
                command = output_feature_labels[i]
                print(f"Confidence above {threshold*100:.0f}% for action '{command}', command will be sent.")
                commands.append((command, True))
            else:
                print(f"Confidence below {threshold*100:.0f}% for action '{output_feature_labels[i]}', no command will be sent.")

        # Prioritize commands if necessary (e.g., forward > left > right > backward)
        if commands:
            # Sort by probability in descending order
            commands = sorted(commands, key=lambda cmd: probabilities[output_feature_labels.index(cmd[0])], reverse=True)
            return [commands[0]]  # Send only the highest-priority command
        return None

    def process_message(self, message, data_collector):
        commands = self.nn_infer(message)
        if commands:
            current_time = time.time()
            command, start = commands[0]
            # If command changes or it’s time to resend due to cooldown
            if (command != self.last_command) or (current_time - self.last_command_time > 0.1):
                self.last_command = command
                self.last_command_time = current_time
                data_collector.onCarControlled(command, start)
            else:
                predicted_action = 'none'


            print(f"Predicted action: {predicted_action}")
            return [(predicted_action, True)]

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

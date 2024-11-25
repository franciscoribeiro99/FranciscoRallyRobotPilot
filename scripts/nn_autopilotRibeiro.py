import sys
import torch
import torch.nn as nn
from PyQt6 import QtWidgets
from models import CNN, CNNColor, ConcatModel
import cv2
from data_collector import DataCollectionUI
import time
import numpy as np

# Load the ConcatModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model.pth"

print("Device:", device)
print("Loading model...")

try:
    cnn_model = CNN(num_classes=4)
    model = ConcatModel(cnn_model, cnn_color).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded and set to evaluation mode.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Define the color to follow
color_to_follow = [255, 0, 0]
print(f"Color to follow: {color_to_follow}")

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

class ConcatModelMsgProcessor:
    def __init__(self):
        self.model = model
        self.last_command = None
        self.last_command_time = 0

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
                # Continuously send "push" if confidence remains high
                data_collector.onCarControlled(command, True)

if __name__ == "__main__":
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    print("Starting application...")

    app = QtWidgets.QApplication(sys.argv)

    print("Initializing neural network message processor...")
    nn_brain = ConcatModelMsgProcessor()

    print("Setting up data collection UI...")
    data_window = DataCollectionUI(nn_brain.process_message)  # Pass the process_message as the callback
    data_window.show()

    print("Running application...")
    app.exec()

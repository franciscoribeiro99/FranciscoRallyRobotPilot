import sys
import torch
import torch.nn as nn
from PyQt6 import QtWidgets

from data_collector import DataCollectionUI
"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows

/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""


model_path = "../rallyBot/models/augmented_nn.pth"
model_dict = torch.load(model_path)


def preprocess_input(s):
    return torch.tensor([s.raycast_distances[3],
    s.raycast_distances[4],
    s.raycast_distances[5],
    s.raycast_distances[6],
    s.raycast_distances[7],
    s.raycast_distances[8],
    s.raycast_distances[9],
    s.raycast_distances[10],
    s.raycast_distances[11]], dtype=torch.float32)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = nn.Softmax(dim=1)(out)
        return out

model = SimpleNN(9, 50, 5)
model.load_state_dict(model_dict)
model.eval()

output_feature_labels = ['forward', 'back', 'left', 'right', 'nothing']



class SimpleNNMsgProcessor:
    def __init__(self):
        self.model = model

    def nn_infer(self, message):
        input_tensor = preprocess_input(message)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if necessary

        with torch.no_grad():
            output = self.model(input_tensor)
        print("Models output: ", output)

        # Process the output to generate commands
        # This part depends on the model's output format
        # For example, if the model outputs probabilities for different actions:
        _, predicted = torch.max(output, 1)
        predicted = predicted.item()
        
        print(output_feature_labels[predicted])

        #must threath return [(output_feature_labels[predicted], True)] like that
        return [(output_feature_labels[predicted], True)]
        

    def process_message(self, message, data_collector):

        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)

if  __name__ == "__main__":
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = SimpleNNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()

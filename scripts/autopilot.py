from PyQt6 import QtWidgets
import torch
import MLP
from ursina import *
from data_collector import DataCollectionUI


class Autopilot:
    def __init__(self):
        model_file = "ModelOld.pth"
        self.model = MLP.MLP()
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()
        self.has_started = False


    def nn_infer(self, message):
        # kickstart the car - if it's not moving, move it
        # i guess doesn't learn to start moving 
        # because most of the recordings at speed 0
        # are not with the accelerator pressed?
        if False and message.car_speed < 1.0:
            self.has_started = True
            return [
                ("forward", True),
                ("back", False),
                ("left", False),
                ("right", False)
            ]

        speed_limiter = message.car_speed > 40.0

        # format the input
        input = list(message.raycast_distances) + [message.car_speed]
        input = self.model.normalize([input])
        input = torch.tensor(input)

        # it expects a 16x7 tensor - so we need to add 0s ?
        input = input.unsqueeze(0)

        # infer
        output = self.model(input)

        forward, back = output[0][0].item(), output[0][1].item()
        left, right = output[0][2].item(), output[0][3].item()

        print(forward, back, left, right)

        go_back, go_forward, go_left, go_right = False, False, False, False
        go_back = back > 0.5
        go_forward = not speed_limiter and not go_back and  forward > 0.5
        go_left = left > 0.5
        go_right = right > 0.5

        if left > 0.5 and right > 0.5:
            if left > right:
                go_right = False
            else:
                go_left = False

        # format the output
        m = [
            ("forward", go_forward),
            ("back", go_back),
            ("left", go_left),
            ("right", go_right) 
        ]

        return m

    def process_message(self, message, data_collector):
        commands = self.nn_infer(message)

        for command, start in commands:
            print("Sending command", command, start)    
            data_collector.onCarControlled(command, start)

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = Autopilot()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()
import sys
import lzma, pickle
from math import sqrt
import time

import base64
import requests
import json

from PyQt6 import QtWidgets

from data_collector import DataCollectionUI

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

    # compute mean distance of
    # outer raycasts
    # outer_raycasts_indexes = [0, 1, 2, 12, 13, 14]
    # car_outer_raycasts = [sum([s.raycast_distances[index] for s in sensing_messages]) / len(sensing_messages) for index in outer_raycasts_indexes]
    # mean_outer_raycasts_distance = sum(car_outer_raycasts) / len(outer_raycasts_indexes)

    print("Metrics computed on CNN-infered car controls:")
    print(f"Mean speed:\t{mean_speed:.3} [units/frame]")
    print(f"Mean acceleration:\t{mean_acceleration:.3} [units/frame²]")
    # print(f"Mean outer raycasts distance:\t{mean_outer_raycasts_distance:.3} [units]")

class Evaluation():
    def __init__(self, *args, **kwargs):
        self.simulation_time = int(args[0][1]) # in seconds
        self.start_time = None

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
                print("[LOG] Total simulation time exceeded. Saving data..")
                if data_collector.saving_worker is None:
                    data_collector.saveRecord()

                    data_collector.recording = False
                    data_collector.message_processing_callback = lambda x, y: None
            else:
                # no timeout issued,
                # controls can be
                # infered from image
                b64_message = base64.b64encode(message.pack())
                ascii_encoded_message = json.dumps(b64_message.decode('ascii'))

                request = requests.post('http://127.0.0.1:5000/api/infer', json={'message': ascii_encoded_message})
                controls = json.loads(request.content)

                for command, start in controls:
                    data_collector.onCarControlled(command, start)

if __name__ == "__main__":
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)

    sys.excepthook = except_hook

    qt_application = QtWidgets.QApplication(sys.argv)
    evaluation = Evaluation(sys.argv)
    data_window = DataCollectionUI(evaluation.handle_message, record=True, onDataSaved=compute_metrics)

    data_window.show()
    qt_application.exec()

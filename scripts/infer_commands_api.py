import sys
import base64

from PyQt6 import QtWidgets
from flask import Flask, jsonify, request
import json

from data_collector import DataCollectionUI
from rallyrobopilot.sensing_message import SensingSnapshot
from cnn_model import CNNMsgProcessor

app = Flask(__name__)

@app.route('/api/infer', methods=['POST'])
def simulate():
    print("[/api/infer/] Received request to simulate")

    data = request.get_json()
    message = data['message']

    decoded_message = base64.b64decode(message)
    unpacked_message = SensingSnapshot()
    unpacked_message.unpack(decoded_message)

    # infer the controls
    # from the sent image
    controls = model.cnn_infer(unpacked_message)

    return jsonify({"controls": json.dumps(controls)})

if __name__ == '__main__':
    model = CNNMsgProcessor()

    app.run(host='0.0.0.0', port=6000, debug=True)

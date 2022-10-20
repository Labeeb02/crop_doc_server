from flask import Flask, request, jsonify

from diseases_predictor import DiseasePredictor
from firebase_services import FirebaseServices
from model import Model

import sys

app = Flask(__name__)
model = Model()


@app.route('/')
def index():
    return "Crop-Doc Disease Prediction and Plotting Server"


@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    if request.method == 'GET':
        return "PREDICTION SERVER"

    if request.method == 'POST':
        image_url = request.json['image_URL']
        selected_crop = request.json['selected_crop']


        disease_predictor = DiseasePredictor(image_url)
        result = disease_predictor.predict_disease(model, selected_crop)

        return jsonify(result)


if __name__ == '__main__':

    firebase = FirebaseServices()
    firebase.update_server_address()

    address = '127.0.0.1'
    port = 5000

    args = sys.argv
    if len(args) == 2:
        address = args[1]
    elif len(args) == 3:
        address = args[1]
        port = args[2]

    app.run(host=address, port=port)

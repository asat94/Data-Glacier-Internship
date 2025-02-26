import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from flask import Flask, request, jsonify
import pickle

import warnings
warnings.filterwarnings("ignore")

with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)


app = Flask(__name__)

@app.route('/')
def home():
    return "Titanic Survival Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"survived": int(prediction[0])})

if __name__ == '__main__':
    app.run(use_reloader=False)


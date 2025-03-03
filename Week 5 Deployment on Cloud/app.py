
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# Load the trained model
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Home route - serves the index.html file
@app.route('/')
def home():
    return render_template('index.html')


# Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['Pclass'], data['Age'], data['Sex'], data['Fare']]])
    prediction = model.predict(features)
    survival = "Survived" if prediction[0] == 1 else "Not Survived"
    return jsonify({"prediction": survival})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug)
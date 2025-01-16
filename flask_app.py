pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org flask

from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enabling Cross-Origin Resource Sharing

# Configuring logging
logging.basicConfig(level=logging.INFO)

# Loading the model and scaler
model = joblib.load(open('air_quality.pkl', 'rb'))
# If you have a scaler, load it as well (uncomment the following line)
scaler = joblib.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return "Air Quality Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting the JSON data from the request
        data = request.get_json()
        logging.info(f"Received data: {data}")

        # Validating input
        required_fields = ['Temperature', 'Humidity', 'PM25', 'PM10', 'SO2', 'NO2', 'CO']
        for field in required_fields:
            if field not in data:
                error_message = f"Missing value for {field}"
                logging.error(error_message)
                return jsonify({"error": error_message}), 400

        # Extracting features from json data
        features = np.array([data['Temperature'], data['Humidity'], data['PM25'], data['PM10'], data['SO2'], data['NO2'], data['CO']]).reshape(1, -1)

        # Scalar transformation
        features = scaler.transform(features)

        # Predictions
        prediction = model.predict(features)
        logging.info(f"Prediction: {prediction[0]}")

        # Returning the prediction as a JSON response
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

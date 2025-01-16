from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('air_quality.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Route for the frontpage
@app.route('/')
def home():
    return render_template('frontpage.html')

# Route for the frontend (predict form)
@app.route('/frontend')
def frontend():
    return render_template('frontend.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    if request.method == 'POST':
            print(request.form)  
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pm25 = float(request.form['pm25'])
            pm10 = float(request.form['pm10'])
            so2 = float(request.form['so2'])
            no2 = float(request.form['no2'])
            co = float(request.form['co'])
            proximity = float(request.form['proximity'])
            population = float(request.form['population'])
            
            pm = pm25 - pm10
            # Create feature array
            features = pd.DataFrame([[temperature, humidity, pm, no2, so2, co, proximity, population]], 
                                columns=['Temperature', 'Humidity', 'PM', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density'])

            
            # Scale features
            features_scaled = scaler.transform(features)
            scaled_features_df = pd.DataFrame(features_scaled, columns=features.columns)

            # Predict using the model
            prediction = model.predict(scaled_features_df)
            prediction = int(prediction[0])
            # Return result as a JSON response
            
            result_message = "Air Quality is Good" if (prediction == 1 )else "Air Quality is Poor"
            print({'Prediction': result_message})
            return jsonify({'Prediction': result_message})
    
            #return render_template('frontend.html', result_message=result_message)

    #return render_template('frontend.html')

if __name__ == "__main__":
    app.run(debug=True)

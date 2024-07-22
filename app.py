from flask import Flask, render_template, request
import numpy as np
import joblib
import logging

app = Flask(__name__)
model = joblib.load('models/crop_recommendation_model.pkl')
scaler = joblib.load('models/scaler.pkl')  # Load the scaler

CROP_DICT = {
    1: 'rice',
    2: 'maize',
    3: 'jute',
    4: 'cotton',
    5: 'coconut',
    6: 'papaya',
    7: 'orange',
    8: 'apple',
    9: 'muskmelon',
    10: 'watermelon',
    11: 'grapes',
    12: 'mango',
    13: 'banana',
    14: 'pomegranate',
    15: 'lentil',
    16: 'blackgram',
    17: 'mungbean',
    18: 'mothbeans',
    19: 'pigeonpeas',
    20: 'kidneybeans',
    21: 'chickpea',
    22: 'coffee'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from form
        data = [float(x) for x in request.form.values()]
        final_features = np.array(data).reshape(1, -1)
        
        # Standardize the inputs
        final_features_scaled = scaler.transform(final_features)
        
        # Predict the crop index
        prediction = model.predict(final_features_scaled)
        
        # Convert numerical prediction to crop name
        crop_name = CROP_DICT.get(prediction[0], "Unknown Crop")
        
        return render_template('index.html', prediction_text=f'Recommended Crop: {crop_name}')
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)


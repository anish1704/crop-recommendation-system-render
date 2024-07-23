from flask import Flask, render_template, request, url_for
import numpy as np
import joblib
import logging
import json

app = Flask(__name__)
model = joblib.load('models/crop_recommendation_model.pkl')
scaler = joblib.load('models/scaler.pkl')  # Load the scaler

# Load crop information from JSON file
with open('data/crop_info.json') as f:
    crop_info = json.load(f)

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
        data = [float(request.form.get(key)) for key in ('N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall')]
        final_features = np.array(data).reshape(1, -1)
        
        # Standardize the inputs
        final_features_scaled = scaler.transform(final_features)
        
        # Predict the crop index
        prediction = model.predict(final_features_scaled)
        
        # Convert numerical prediction to crop name
        crop_id = prediction[0]
        crop_name = CROP_DICT.get(crop_id, "Unknown Crop")
        
        # Get crop information
        crop_details = crop_info.get(crop_name, {})
        image_file = crop_details.get("crop_image", "default.jpg")
        
        prediction_data = {
            'crop_name': crop_name,
            'image': image_file,
            'info': crop_details
        }
        
        return render_template('index.html', prediction=prediction_data)
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)

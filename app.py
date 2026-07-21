import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form input in exact order expected by the model
        features = [
            float(request.form['bedrooms']),
            float(request.form['bathrooms']),
            float(request.form['living_area']),
            float(request.form['lot_area']),
            float(request.form['floors']),
            float(request.form['waterfront']),
            float(request.form['views']),
            float(request.form['condition']),
            float(request.form['grade']),
            float(request.form['area_no_basement']),
            float(request.form['basement_area']),
            float(request.form['built_year']),
            float(request.form['renovation_year']),
            float(request.form['lot_area_renov']),
            float(request.form['schools']),
            float(request.form['airport_dist'])
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        formatted_price = f"${prediction:,.2f}"
        
        return render_template('index.html', prediction_text=f"Estimated House Value: {formatted_price}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error processing input: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

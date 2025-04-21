from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("wine_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from the form
        features = [float(x) for x in request.form.values()]
        features_array = np.array([features])
        
        # Scale the input features just like training
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return render_template('index.html', prediction_text=f'Predicted Wine Quality: {prediction:.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)

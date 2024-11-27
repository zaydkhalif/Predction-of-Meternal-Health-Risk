from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler using joblib
model = joblib.load('KNeighborsClassifier.pkl')
scaler = joblib.load('scaler.pkl')

# This function uses the loaded model and scaler for risk prediction
def predict_risk(age, diastolic_bp, glucose, temperature, heart_rate):
    # Prepare the input data in the format the model expects (2D array)
    input_data = np.array([[age, diastolic_bp, glucose, temperature, heart_rate]])
    
    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using the loaded model
    prediction = model.predict(input_data_scaled)
    
    # Map the prediction to a risk level
    if prediction[0] == 0:
        return "Low Risk Level"
    elif prediction[0] == 1:
        return "Mid Risk Level"
    else:
        return "High Risk Level"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        age = int(request.form['age'])
        diastolic_bp = int(request.form['diastolic_bp'])
        glucose = float(request.form['glucose'])
        temperature = float(request.form['temperature'])
        heart_rate = int(request.form['heart_rate'])

        # Predict risk level using the model
        risk_level = predict_risk(age, diastolic_bp, glucose, temperature, heart_rate)
        
        # Render the result back to the page
        return render_template('index.html', risk_level=risk_level)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

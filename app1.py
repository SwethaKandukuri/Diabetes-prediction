from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

# Data Preparation
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardization
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Home route - displays the form
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    if request.method == 'POST':
        # Get user input from the form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = float(request.form['age'])

        # Create input data array
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)

        # Standardize the input data
        std_data = scaler.transform(input_data)

        # Make prediction
        prediction = classifier.predict(std_data)
        prediction_result = 'The person is not diabetic' if prediction[0] == 0 else 'The person is diabetic'

    # Render the HTML template with the form and prediction result
    return render_template('index.html', prediction_result=prediction_result)

# Run the Flask app
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Use Render's port or default to 5000
    app.run(host='0.0.0.0', port=port)

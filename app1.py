from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import os
from joblib import dump, load  # NEW: For model caching

# Initialize Flask app
app = Flask(__name__)

# NEW: Constants for validation
MIN_MAX_VALUES = {
    'pregnancies': (0, 20),
    'glucose': (50, 300),
    'blood_pressure': (30, 150),
    'skin_thickness': (0, 99),
    'insulin': (0, 900),
    'bmi': (10, 60),
    'diabetes_pedigree': (0, 2.5),
    'age': (1, 120)
}

# NEW: Load or train model
def load_or_train_model():
    model_path = os.path.join(os.path.dirname(__file__), 'diabetes_model.joblib')
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        return load(model_path)
    else:
        print("Training new model...")
        # Load and prepare data
        csv_path = os.path.join(os.path.dirname(__file__), 'diabetes.csv')
        diabetes_dataset = pd.read_csv(csv_path)
        X = diabetes_dataset.drop(columns='Outcome', axis=1)
        Y = diabetes_dataset['Outcome']
        
        # Standardization
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        # Train-Test Split
        X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
        
        # Model Training
        classifier = svm.SVC(kernel='linear', probability=True)  # NEW: Added probability=True for confidence scores
        classifier.fit(X_train, Y_train)
        
        # Save model for future runs
        dump(classifier, model_path)
        return classifier

# Load model and scaler
classifier = load_or_train_model()

# NEW: Load scaler separately (you'll need to save this after first run)
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
if os.path.exists(scaler_path):
    scaler = load(scaler_path)
else:
    # Initial scaler setup (only runs once)
    csv_path = os.path.join(os.path.dirname(__file__), 'diabetes.csv')
    diabetes_dataset = pd.read_csv(csv_path)
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    dump(scaler, scaler_path)

# Home route - displays the form
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    error = None  # NEW: For error messages
    
    if request.method == 'POST':
        try:
            # NEW: Input validation
            input_values = {}
            for field in MIN_MAX_VALUES:
                value = float(request.form[field])
                min_val, max_val = MIN_MAX_VALUES[field]
                
                if not (min_val <= value <= max_val):
                    error = f"Invalid {field.replace('_', ' ')}: must be between {min_val}-{max_val}"
                    break
                
                input_values[field] = value
            
            if not error:
                # Create input data array
                input_data = np.array([
                    input_values['pregnancies'],
                    input_values['glucose'],
                    input_values['blood_pressure'],
                    input_values['skin_thickness'],
                    input_values['insulin'],
                    input_values['bmi'],
                    input_values['diabetes_pedigree'],
                    input_values['age']
                ]).reshape(1, -1)

                # Standardize the input data
                std_data = scaler.transform(input_data)

                # NEW: Get prediction with confidence score
                prediction = classifier.predict(std_data)
                proba = classifier.predict_proba(std_data)[0]
                confidence = round(max(proba) * 100, 2)
                
                # NEW: Enhanced result format
                if prediction[0] == 1:
                    prediction_result = f'Diabetic ({confidence}% confidence)'
                else:
                    prediction_result = f'Not Diabetic ({confidence}% confidence)'
                    
        except ValueError:
            error = "Please enter valid numbers in all fields"
        except Exception as e:
            error = f"An error occurred: {str(e)}"

    # Render the HTML template
    return render_template(
        'index.html',
        prediction_result=prediction_result,
        error=error  # NEW: Pass error to template
    )

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
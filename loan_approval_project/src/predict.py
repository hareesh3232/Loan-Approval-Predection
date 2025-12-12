import pandas as pd
import joblib
import os
import numpy as np

MODELS_DIR = os.path.join('loan_approval_project', 'models')

def load_artifacts():
    model_path = os.path.join(MODELS_DIR, 'loan_model.pkl')
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model or Preprocessor not found. Run training first.")
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def predict_loan(input_data):
    """
    input_data: dict containing feature values
    """
    model, preprocessor = load_artifacts()
    
    # Convert dict to DataFrame
    df = pd.DataFrame([input_data])
    
    # Preprocess
    # Ensure columns match what preprocessor expects
    # The preprocessor was fitted on specific columns.
    # Note: If Credit_History is passed as 1.0 or 0.0, ensure type consistency.
    
    X_processed = preprocessor.transform(df)
    
    # Predict
    prediction = model.predict(X_processed)
    probability = model.predict_proba(X_processed)[:, 1] if hasattr(model, "predict_proba") else None
    
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    return result, probability[0] if probability is not None else 0.0

if __name__ == "__main__":
    # Example usage
    sample_input = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }
    
    print("Predicting for sample input:", sample_input)
    try:
        status, prob = predict_loan(sample_input)
        print(f"Prediction: {status}")
        print(f"Probability: {prob:.2f}")
    except Exception as e:
        print(f"Error: {e}")

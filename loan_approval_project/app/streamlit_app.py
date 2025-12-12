import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Add src to path to import predict logic if needed, or just load directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Define paths
MODEL_PATH = os.path.join('loan_approval_project', 'models', 'loan_model.pkl')
PREPROCESSOR_PATH = os.path.join('loan_approval_project', 'models', 'preprocessor.pkl')

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="centered")

st.title("💰 Loan Approval Prediction System")
st.markdown("Enter the applicant details below to check loan eligibility.")

@st.cache_resource
def load_model_and_preprocessor():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

if model is None:
    st.error("Model or Preprocessor not found! Please run `src/train_model.py` first.")
else:
    # Input Form
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            credit_history_label = st.selectbox("Credit History", ["Good (1.0)", "Bad (0.0)"])
        
        with col2:
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
            loan_amount_term = st.selectbox("Loan Amount Term (Months)", [360, 180, 120, 84, 60, 300, 480])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            
        credit_history = 1.0 if "1.0" in credit_history_label else 0.0
        
        submit_button = st.form_submit_button("Predict Loan Status")
        
        if submit_button:
            # Prepare Input
            input_data = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': property_area
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Processing
            try:
                X_processed = preprocessor.transform(input_df)
                prediction = model.predict(X_processed)
                probability = model.predict_proba(X_processed)[:, 1] if hasattr(model, "predict_proba") else None
                
                st.markdown("---")
                if prediction[0] == 1:
                    st.success("✅ **Loan Approved!**")
                    if probability is not None:
                        st.info(f"Confidence: {probability[0]*100:.2f}%")
                else:
                    st.error("❌ **Loan Not Approved**")
                    if probability is not None:
                        st.info(f"Confidence: {(1-probability[0])*100:.2f}%")
                        
            except Exception as e:
                st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Machine Learning Project - Loan Prediction")

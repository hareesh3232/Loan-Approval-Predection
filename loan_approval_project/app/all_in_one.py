import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Loan Prediction All-in-One", page_icon="🏦", layout="wide")

# File Paths
DATA_FILE = os.path.join('loan_approval_project', 'data', 'loan_prediction.csv')

# Ensure directories exist
os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

# ==========================================
# 2. DATA GENERATION / LOADING
# ==========================================
def get_data():
    if not os.path.exists(DATA_FILE):
        st.warning("⚠️ Dataset not found. Generating synthetic data...")
        # Synthetic Data Generation Logic
        num_rows = 600
        data = {
            'Loan_ID': [f'LP00{i}' for i in range(1000, 1000 + num_rows)],
            'Gender': np.random.choice(['Male', 'Female', np.nan], size=num_rows, p=[0.75, 0.2, 0.05]),
            'Married': np.random.choice(['Yes', 'No', np.nan], size=num_rows, p=[0.6, 0.38, 0.02]),
            'Dependents': np.random.choice(['0', '1', '2', '3+', np.nan], size=num_rows, p=[0.5, 0.2, 0.2, 0.08, 0.02]),
            'Education': np.random.choice(['Graduate', 'Not Graduate'], size=num_rows, p=[0.8, 0.2]),
            'Self_Employed': np.random.choice(['Yes', 'No', np.nan], size=num_rows, p=[0.15, 0.8, 0.05]),
            'ApplicantIncome': np.random.randint(1500, 80000, size=num_rows),
            'CoapplicantIncome': np.random.randint(0, 40000, size=num_rows),
            'LoanAmount': np.random.randint(50, 700, size=num_rows),
            'Loan_Amount_Term': np.random.choice([360, 180, 480, 300, np.nan], size=num_rows, p=[0.8, 0.1, 0.05, 0.02, 0.03]),
            'Credit_History': np.random.choice([1.0, 0.0, np.nan], size=num_rows, p=[0.8, 0.15, 0.05]),
            'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], size=num_rows),
        }
        df = pd.DataFrame(data)
        
        # Simple logic for target
        def get_status(row):
            score = 0
            if row['Credit_History'] == 1.0: score += 5
            if row['Education'] == 'Graduate': score += 1
            if row['Married'] == 'Yes': score += 1
            if row['ApplicantIncome'] > 5000: score += 2
            score += np.random.randint(-2, 3) 
            return 'Y' if score > 4 else 'N'

        df['Loan_Status'] = df.apply(get_status, axis=1)
        df.to_csv(DATA_FILE, index=False)
        st.success("✅ Synthetic data generated!")
    
    return pd.read_csv(DATA_FILE)

# ==========================================
# 3. TRAINING PIPELINE (Cached)
# ==========================================
@st.cache_resource
def train_model():
    df = get_data()
    
    # Preprocessing
    df = df.drop_duplicates()
    X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y) # N=0, Y=1
    
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    credit_history_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('credit', credit_history_transformer, ['Credit_History'])
        ])
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create Full Pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Train
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return clf, acc, f1, le

# ==========================================
# 4. FRONTEND / UI
# ==========================================

st.title("🏦 Loan Approval : All-in-One App")
st.markdown("This single application handles **Data Loading**, **Model Training**, **Evaluation**, and **Prediction**.")

# -----------------
# Section: Training
# -----------------
with st.expander("📊 Model Status & Performance", expanded=True):
    with st.spinner('Training model... please wait...'):
        pipeline, acc, f1, le = train_model()
    
    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy", f"{acc*100:.2f}%")
    col2.metric("F1 Score", f"{f1:.2f}")
    st.caption("Model: Random Forest Classifier")

# -----------------
# Section: Prediction
# -----------------
st.divider()
st.header("🔮 Make a Prediction")

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
    loan_amount = st.number_input("Loan Amount", min_value=0, value=120)
    loan_amount_term = st.selectbox("Loan Amount Term (Months)", [360, 180, 120, 84, 60, 300, 480])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

credit_history = 1.0 if "1.0" in credit_history_label else 0.0

if st.button("Check Loan Eligibility", type="primary"):
    # Create DF for prediction
    input_data = {
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    }
    input_df = pd.DataFrame(input_data)
    
    # Predict
    prediction_idx = pipeline.predict(input_df)[0]
    prediction_prob = pipeline.predict_proba(input_df)[0][1]
    
    prediction_label = le.inverse_transform([prediction_idx])[0]
    
    st.markdown("### Result")
    if prediction_label == 'Y':
        st.success(f"✅ **Loan Approved** (Probability: {prediction_prob:.2%})")
    else:
        st.error(f"❌ **Loan Rejected** (Probability: {1-prediction_prob:.2%})")

st.markdown("---")
st.markdown("Designed with Streamlit & Scikit-Learn")

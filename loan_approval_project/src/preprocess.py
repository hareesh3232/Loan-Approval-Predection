import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Paths
# Paths relative to project root
DATA_PATH = os.path.join('data', 'loan_prediction.csv')
PROCESSED_DATA_DIR = os.path.join('data')
MODELS_DIR = os.path.join('models')

# Create dirs if not exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def preprocess_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Handle Duplicates
    df = df.drop_duplicates()
    
    # 2. Separate Target
    # Prepare X and y
    # Drop Loan_ID as it is not a feature
    X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status']
    
    # Encode Target
    le = LabelEncoder()
    y = le.fit_transform(y) # Y->1, N->0 usually, check classes
    print(f"Target Classes: {le.classes_}") # ['N' 'Y'] -> 0, 1
    
    # Save LabelEncoder for target if needed (simple enough to remember 1=Approved)
    
    # 3. Define Features
    # Numeric features
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    # Categorical features
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    # Credit History is a special case, treat as categorical or binary. 
    # Since it is 0.0 or 1.0, we can treat it as numeric (binary) but it has missing values.
    # Let's add it to numeric for simple imputation with mode or categorical using 'most_frequent'
    
    # Actually Credit_History is extremely important, let's treat it as a separate group for 'most_frequent' imputation
    # but not scaling (it's 0/1). For logistics regression, 0/1 is fine.
    
    # Updated Categorical List to include Credit_History if we want to OneHot it? 
    # No, it's binary. Let's just impute it.
    
    # Let's split simple numeric vs categorical text
    
    # Pipelines
    
    # Numeric Pipeline: Impute Median -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical Pipeline: Impute Mode -> OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Credit History Pipeline: Impute Mode (don't scale 0/1)
    # Note: Credit_History in dataframe is float with NaNs.
    credit_history_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('credit', credit_history_transformer, ['Credit_History'])
        ])
    
    # 4. Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Fit Preprocessor on Train
    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after OneHot (helpful for feature importance)
    # This might be tricky with ColumnTransformer in older sklearn versions, but let's try generic approach if needed later.
    
    # 6. Save Processed Data
    # Saving as numpy arrays or dataframe? Arrays are safer for sklearn but lose columns.
    # Let's save as numpy arrays (npy) or just CSVs (without headers is messy)
    # Let's save as npz or pandas df? 
    # For transparency, let's try to reconstruct dataframe headers, but it's complex.
    # The prompt asks to "Save processed datasets". 
    # I will save as .npy files for robustness in loading during training.
    
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train_processed)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test_processed)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    # Save the preprocessor pipeline
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    print("Preprocessor saved to models/preprocessor.pkl")
    print("Processed data saved to data/")

if __name__ == "__main__":
    preprocess_data()

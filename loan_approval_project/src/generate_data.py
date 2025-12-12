import pandas as pd
import numpy as np
import os
import random

def generate_data():
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
    
    # Introduce some logical correlation for Loan_Status
    # High credit history -> likely approved
    # Low income + High Loan -> likely rejected
    
    df = pd.DataFrame(data)
    
    def get_status(row):
        score = 0
        if row['Credit_History'] == 1.0:
            score += 5
        if row['Education'] == 'Graduate':
            score += 1
        if row['Married'] == 'Yes':
            score += 1
        if row['ApplicantIncome'] > 5000:
            score += 2
        
        # Randomness
        score += np.random.randint(-2, 3)
        
        return 'Y' if score > 4 else 'N'

    df['Loan_Status'] = df.apply(get_status, axis=1)
    
    # Inject missing values for LoanAmount (simulate real data issues)
    df.loc[np.random.choice(df.index, 20), 'LoanAmount'] = np.nan

    os.makedirs('loan_approval_project/data', exist_ok=True)
    df.to_csv('loan_approval_project/data/loan_prediction.csv', index=False)
    print("Synthetic dataset created at loan_approval_project/data/loan_prediction.csv")

if __name__ == "__main__":
    generate_data()

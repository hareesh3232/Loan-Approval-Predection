# Loan Approval Prediction

## Project Overview
This project is an end-to-end Machine Learning system to predict loan approval status based on applicant details. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and a Streamlit web application for real-time predictions.

## Dataset Description
The dataset selected for this project is the **Loan Prediction Dataset**. It contains data about loan applicants and their approval status.

### Features:
| Feature | Description |
| :--- | :--- |
| **Loan_ID** | Unique Loan ID |
| **Gender** | Male/Female |
| **Married** | Applicant married (Y/N) |
| **Dependents** | Number of dependents |
| **Education** | Graduate/Under Graduate |
| **Self_Employed** | Self employed (Y/N) |
| **ApplicantIncome** | Applicant income |
| **CoapplicantIncome** | Coapplicant income |
| **LoanAmount** | Loan amount in thousands |
| **Loan_Amount_Term** | Term of loan in months |
| **Credit_History** | Credit history meets guidelines (1.0/0.0) |
| **Property_Area** | Urban/Semi Urban/Rural |
| **Loan_Status** | Loan Approved (Y/N) **[Target Variable]** |

## Model Explanation
We evaluated multiple Machine Learning algorithms to find the best fit for this loan approval task:
1.  **Logistic Regression**: A statistical model used for binary classification. It works well when the relationship between features and the log-odds of the outcome is linear.
2.  **Random Forest Classifier**: An ensemble learning method based on decision trees. It handles non-linear relationships well and is robust to overfitting.
3.  **XGBoost**: An optimized gradient boosting algorithm known for its high performance and speed.

**Selected Model:**
Based on our evaluation, **Logistic Regression** achieved the best balance of Accuracy and F1 Score for this specific dataset and preprocessing pipeline.

## Evaluation Metrics
The models were evaluated using Accuracy, Precision, Recall, and F1 Score. The results on the test set are as follows:

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **86.18%** | 0.84 | 0.99 | **0.91** |
| Random Forest | 82.93% | 0.85 | 0.92 | 0.88 |
| XGBoost | 78.86% | 0.84 | 0.86 | 0.85 |

*Note: The high recall of Logistic Regression indicates it is very good at identifying positive loan cases, which is crucial for maximizing business opportunity, though care must be taken with false positives.*

## Technologies Used
- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation.
- **Scikit-Learn**: Machine learning models and preprocessing.
- **XGBoost**: Gradient boosting classifier.
- **Matplotlib & Seaborn**: Data visualization.
- **Streamlit**: Web interface for the application.

## Project Structure
```
loan_approval_project/
├── data/               # Dataset files
├── src/                # Source code for preprocessing, training, etc.
├── models/             # Saved trained models
├── app/                # Streamlit application
├── notebooks/          # Jupyter notebooks for EDA
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate/Place Data
Ensure `loan_prediction.csv` is in the `data/` folder.
*If you need to generate synthetic data for testing, run the provided helper script (if applicable) or use the Kaggle 'Loan Prediction Problem' dataset.*

### 3. Run Preprocessing
```bash
python src/preprocess.py
```
This will clean the data and save train/test sets to `data/`.

### 4. Train Model
```bash
python src/train_model.py
```
This trains the models and saves the best one to `models/loan_model.pkl`.

### 5. Evaluate Model
```bash
python src/evaluate.py
```
Generates evaluation metrics and plots.

### 6. Run Prediction Script
```bash
python src/predict.py
```

### 7. Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Example Input/Output
**Input:**
- Gender: Male
- Married: Yes
- Income: 5000
- Loan Amount: 120
- Credit History: 1.0

**Output:**
- Loan Approved (or Not Approved)

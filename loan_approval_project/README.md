# Loan Approval Prediction

## Project Overview
This project is an end-to-end Machine Learning system to predict loan approval status based on applicant details. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and a Streamlit web application for real-time predictions.

## Dataset
The dataset used is `loan_prediction.csv`. It contains columns like Gender, Marital Status, Education, Income, Loan Amount, Credit History, etc.

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

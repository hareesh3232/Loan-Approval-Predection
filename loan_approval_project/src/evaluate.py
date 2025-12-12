import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

DATA_DIR = os.path.join('loan_approval_project', 'data')
MODELS_DIR = os.path.join('loan_approval_project', 'models')
PLOTS_DIR = os.path.join('loan_approval_project', 'plots')

os.makedirs(PLOTS_DIR, exist_ok=True)

def evaluate_model():
    print("Loading data and model...")
    # Load data
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    
    # Load model
    model_path = os.path.join(MODELS_DIR, 'loan_model.pkl')
    if not os.path.exists(model_path):
        print("Model not found. Please run train_model.py first.")
        return
        
    model = joblib.load(model_path)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
    print("Confusion Matrix saved.")
    
    # 2. Classification Report
    report = classification_report(y_test, y_pred, target_names=['Rejected', 'Approved'])
    print("\nClassification Report:\n")
    print(report)
    with open(os.path.join(PLOTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    # 3. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'))
    print("ROC Curve saved.")

if __name__ == "__main__":
    evaluate_model()

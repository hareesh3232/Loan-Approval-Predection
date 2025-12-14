import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False
    print("XGBoost not installed, skipping.")

# Define paths relative to the project root
# Assuming this script is run from the project root (where src/ is)
DATA_DIR = os.path.join('data')
MODELS_DIR = os.path.join('models')

def train_models():
    print("Loading prepared data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy')) # Used for final selection validation if needed, but better to use CV
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    if xgb_available:
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    best_model = None
    best_f1 = 0.0
    best_name = ""
    
    results = []

    print("Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        print(f"Model: {name} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
            
    # Save best model
    if best_model:
        save_path = os.path.join(MODELS_DIR, 'loan_model.pkl')
        joblib.dump(best_model, save_path)
        print(f"\nBest Model: {best_name} with F1: {best_f1:.4f}")
        print(f"Model saved to {save_path}")
        
        # Save performance report
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(MODELS_DIR, 'model_performance.csv'), index=False)
        print(results_df)
    else:
        print("Model training failed.")

if __name__ == "__main__":
    train_models()

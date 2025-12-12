import logging
import os
import joblib

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"Model not found at {path}")

import joblib
import os

def load_model(path=None):
    if path is None:
        path = os.path.join("..", "models", "xgb_model.joblib")
    return joblib.load(path)
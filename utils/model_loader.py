import joblib
import os

def load_model(path=None):
    if path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "..", "models", "xgb_model.joblib")
    return joblib.load(path)
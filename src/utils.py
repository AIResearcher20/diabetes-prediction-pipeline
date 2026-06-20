"""
utils.py
Helper functions for saving and loading artifacts.
"""
import joblib
import os

def save_model(model, path="models/best_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def load_model(path="models/best_model.pkl"):
    return joblib.load(path)

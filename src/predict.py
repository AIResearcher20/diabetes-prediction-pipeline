"""
predict.py
Prediction utility for new samples.
"""
import numpy as np

def predict_diabetes(model, scaler, sample):
    """
    Predict diabetes risk for a single sample.
    
    Args:
        model: Trained Random Forest model
        scaler: Fitted StandardScaler
        sample: List or array of 8 features
    
    Returns:
        dict: Prediction (0/1) and probability
    """
    sample_scaled = scaler.transform(np.array(sample).reshape(1, -1))
    pred = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0][1]
    
    return {
        "prediction": int(pred),
        "probability": float(prob)
    }

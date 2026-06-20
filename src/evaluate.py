"""
evaluate.py
Model evaluation and metric reporting.
"""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"])
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }

def compare_models(models, X_train, y_train, X_test, y_test):
    """
    Train and compare multiple models.
    """
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({
            "Model": name,
            "Test Accuracy": acc
        })
    return pd.DataFrame(results)

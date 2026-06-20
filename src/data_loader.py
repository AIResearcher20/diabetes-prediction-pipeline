"""
data_loader.py
Loads and validates the PIMA Diabetes dataset.
"""
import pandas as pd
import urllib.request
import os

def load_data(url=None, local_path=None):
    """
    Load PIMA Diabetes dataset from URL or local file.
    
    Args:
        url (str): URL to download dataset (default: PIMA raw)
        local_path (str): Local CSV path if already downloaded
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if local_path and os.path.exists(local_path):
        return pd.read_csv(local_path)
    
    if url is None:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    
    df = pd.read_csv(url, names=columns)
    return df

def get_features_and_target(df, target_column="Outcome"):
    """
    Split dataframe into features (X) and target (y).
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

"""
preprocess.py
Data preprocessing and scaling utilities.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scale_features(X_train, X_test):
    """
    Fit StandardScaler on training data and transform both train and test.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with reproducibility.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

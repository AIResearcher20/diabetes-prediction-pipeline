#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py
Entry point for the diabetes prediction pipeline.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data, get_features_and_target
from src.preprocess import train_test_split_data, scale_features
from src.train_model import train_random_forest
from src.evaluate import evaluate_model, compare_models
from src.utils import save_model

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def main():
    print("="*60)
    print("🩺 Diabetes Risk Prediction Pipeline")
    print("="*60)
    
    # 1. Load data
    print("\n📥 Loading dataset...")
    df = load_data()
    X, y = get_features_and_target(df)
    print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    # 3. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 4. Train and optimize Random Forest
    print("\n🔧 Training Random Forest with hyperparameter tuning...")
    best_model, best_params, best_cv = train_random_forest(X_train_scaled, y_train)
    print(f"✅ Best Parameters: {best_params}")
    print(f"✅ Best CV Accuracy: {best_cv:.4f}")
    
    # 5. Evaluate
    print("\n📊 Evaluating model on test set...")
    results = evaluate_model(best_model, X_test_scaled, y_test)
    print(f"✅ Test Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['report'])
    
    # 6. Cross-validation
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
    print(f"\n✅ 5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 7. Compare models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    comparison_df = compare_models(models, X_train_scaled, y_train, X_test_scaled, y_test)
    print("\n📊 Model Comparison:")
    print(comparison_df.to_string(index=False))
    
    # 8. Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Feature Importance:")
    print(importance.to_string(index=False))
    
    # 9. Save model
    save_model(best_model)
    
    # 10. Plot importance
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance['feature'], importance['importance'], color='steelblue')
    plt.xlabel('Importance Score')
    plt.title('Feature Importance for Diabetes Prediction')
    for bar, imp in zip(bars, importance['importance']):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{imp*100:.1f}%', va='center', fontsize=10)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/feature_importance.png", dpi=300)
    plt.show()
    
    print("\n" + "="*60)
    print("✅ Pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()

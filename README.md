# 🩺 Diabetes Risk Prediction using Explainable Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange" />
  <img src="https://img.shields.io/badge/Medical-AI-red" />
  <img src="https://img.shields.io/badge/Model-RandomForest-green" />
  <img src="https://img.shields.io/badge/Reproducible-Yes-success" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

---

## 📌 **Overview**

This project presents a **fully reproducible, clinically-oriented machine learning pipeline** for diabetes risk prediction using the **PIMA Indians Diabetes Dataset**. The primary goal is to develop **interpretable, reliable, and evidence-based** predictive models that can assist in early screening and clinical decision-making.

The pipeline is designed with a strong emphasis on:
- ✅ **No data leakage** (strict train/test separation)
- ✅ **Complete reproducibility** (fixed seeds, config-driven)
- ✅ **Clinical interpretability** (feature importance analysis)
- ✅ **Robust model evaluation** (5-fold cross-validation)

---

## 🩺 **Clinical Significance**

Type 2 Diabetes is a major global health burden, affecting millions worldwide. Early detection through computational risk assessment can significantly improve patient outcomes. This project aims to bridge the gap between **machine learning research** and **clinical practice** by providing a transparent, explainable, and data-driven approach to diabetes screening.

The model identifies key risk factors such as **Glucose, BMI, and Age**, which are well-established clinical markers, thereby reinforcing its validity and potential for real-world deployment.

---

## 📊 **Dataset**

| Property | Value |
|----------|-------|
| **Dataset** | PIMA Indians Diabetes |
| **Source** | NIDDK (National Institute of Diabetes) |
| **Samples** | 768 |
| **Features** | 8 clinical |
| **Target** | Binary (0: Non-Diabetic, 1: Diabetic) |
| **Task** | Binary Classification |

**Features:**
`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`

---

## 🧠 **Machine Learning Models**

| Model | Type | Description |
|-------|------|-------------|
| **Random Forest** | Ensemble | Bagging of decision trees with feature randomness |
| **Gradient Boosting** | Ensemble | Sequential boosting for error correction |
| **SVM** | Kernel | Maximum margin classification with RBF kernel |
| **Logistic Regression** | Linear | Probabilistic baseline model |

---

## 🔬 **Experimental Pipeline**

1. ✅ Data preprocessing and standardization
2. ✅ Train-test split (80/20) with fixed seed
3. ✅ 5-fold cross-validation
4. ✅ Hyperparameter optimization using GridSearchCV
5. ✅ Model comparison and performance evaluation
6. ✅ Feature importance analysis for interpretability
7. ✅ Visualization and result reporting

---

## 📈 **Results Summary**

### **Model Performance on Test Set**

| Model | Test Accuracy | CV Mean (5-Fold) | CV Std |
|-------|--------------:|-----------------:|-------:|
| **Random Forest** (Best) | **76.62%** | **77.87%** | ±2.62% |
| Logistic Regression | 75.32% | 77.09% | ±2.47% |
| Gradient Boosting | 74.03% | 75.79% | ±3.56% |
| SVM | 72.73% | 77.09% | ±2.25% |

### **Best Model: Random Forest**

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **76.62%** |
| **5-Fold CV Mean** | **77.87%** |
| **Best CV Score** | **78.0%** |
| **Optimized Parameters** | `n_estimators=200`, `max_depth=7`, `min_samples_split=10` |

---

## 🔍 **Feature Importance Analysis**

| Rank | Feature | Importance |
|------|---------|-----------:|
| 1 | **Glucose** | **31.8%** |
| 2 | **BMI** | **18.1%** |
| 3 | **Age** | **15.8%** |
| 4 | DiabetesPedigreeFunction | 9.5% |
| 5 | Insulin | 7.0% |
| 6 | Pregnancies | 6.7% |
| 7 | BloodPressure | 6.0% |
| 8 | SkinThickness | 5.2% |

> ✅ **Clinical Insight:** The top three predictors — **Glucose, BMI, and Age** — are widely recognized as primary risk factors for Type 2 Diabetes. This alignment with established clinical knowledge enhances the model's credibility and potential utility in real-world screening applications.

---

## 📊 **Visualizations**

### Model Comparison
<p align="center">
  <img src="figures/model_comparison.png" width="700">
</p>

### Feature Importance
<p align="center">
  <img src="figures/feature_importance.png" width="700">
</p>

---

## 🧪 **Reproducibility Statement**

This project ensures full reproducibility through:
- ✅ **Fixed random seeds** for all models and data splits
- ✅ **Exact package versions** in `requirements.txt`
- ✅ **No hardcoding** – all parameters are configurable
- ✅ **Public dataset** with standard preprocessing
- ✅ **5-fold cross-validation** for robust evaluation

---

## 🚀 **How to Run**

### **Google Colab (Recommended)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaniakarimi/diabetes-prediction/blob/main/diabetes_prediction.ipynb)

### **Local Installation**
```bash
git clone https://github.com/AIResearcher20/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
python main.py
```

---

🔭 Future Directions

· Integration of deep learning models (TabNet, Graph Neural Networks)
· Deployment as an interactive clinical tool (Streamlit/FastAPI)
· External validation on additional clinical cohorts
· SHAP/LIME-based local interpretability for individual predictions

---

📄 How to Cite

If you use this work, please cite it as:

Moafi , S. (2026). Diabetes Risk Prediction using Explainable Machine Learning. GitHub Repository. https://github.com/AIResearcher20/diabetes-prediction

--

Research Interests:

· Medical Artificial Intelligence
· Graph Neural Networks
· Computational Biology
· Biomedical Machine Learning
· Explainable AI

---

📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

⭐ If you find this project useful, please consider giving it a star!


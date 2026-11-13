# 🛡️ Mobile-Wallet Fraud Detection & Cost-Based Threshold Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM%20%7C%20Random%20Forest-success)
![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-FinTech%20%7C%20Payments-informational)

An end-to-end Machine Learning system designed to detect fraudulent transfers and cash-outs across millions of highly imbalanced mobile-money transactions (modeled after services like **Vodafone Cash** and **InstaPay**), paired with an interactive **Streamlit** decision-support dashboard.

---

## 📌 Problem Statement & Business Impact
Mobile-money platforms process millions of transactions daily where fraud accounts for less than **0.15%** of total volume. A naive model optimizes accuracy at the expense of missing costly fraud, whereas an overly aggressive rule blocks legitimate users and drives customer churn.

This project solves both challenges by:
1. Scaling data processing and feature engineering across **~6.3 million transactions** (PaySim dataset).
2. Optimizing **Precision-Recall AUC (PR-AUC)** under extreme class imbalance.
3. Implementing a **Financial Cost Matrix** to tune the classification threshold that minimizes total monetary loss (`Missed Fraud Cost` vs. `Blocked Honest User Cost`).

---

## 📂 Repository Structure
```text
├── Mobile_Wallet_Fraud_Detection.ipynb   # Full EDA, feature engineering, modeling & threshold tuning
├── app.py                                # Interactive Streamlit web application
├── fraud_model_bundle.joblib             # Serialized model, feature schema & optimal threshold
├── sample_transactions.csv               # Sample test transactions for batch CSV scoring
└── requirements.txt                      # Python dependencies

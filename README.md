# 🛡️ Mobile-Wallet Fraud Detection & Cost-Based Threshold Tuner

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

An end-to-end Machine Learning system and interactive web application designed to detect fraudulent mobile-wallet transactions in real time. Unlike traditional models that rely solely on static accuracy metrics, this solution incorporates a **Cost-Based Decision Threshold Tuner** to balance the financial impact of missed fraud against customer friction from false alarms.

---

## 🌟 Key Highlights & Engineering Decisions

* **Zero Data Leakage Architecture:** The model strictly utilizes **pre-transaction features** available at the exact moment a transaction is initiated (excluding post-transaction balances), ensuring realistic production performance.
* **Imbalanced Data Handling:** Trained using a **Random Forest classifier with 10:1 undersampling**, achieving an exceptional **Test PR-AUC of 1.0000 (95% CI: 1.000–1.000)**.
* **Dynamic Cost-Matrix Optimization:** Allows risk analysts to adjust the classification threshold (default: `0.69`) and customize business costs—**False Alarm Cost** (e.g., `$15.00` per blocked legitimate user) vs. **Missed Fraud Cost** (e.g., `$500.00` per undetected fraud)—to find the financially optimal operating point.
* **Dual Evaluation Modes:**
  1. **Single Transaction Scoring:** Instant risk scoring for individual transactions based on transaction type, amount, time step, and pre-transaction sender/recipient balances.
  2. **Batch CSV Upload & Cost Simulator:** Bulk inference and financial impact simulation across historical or batch transaction logs.

---

## 📂 Repository Structure

```text
├── .devcontainer/                         # Dev container configuration for reproducible environments
├── artifacts/                             # Saved model binaries, scalers, and threshold metadata
├── Mobile_Wallet_Fraud_Detection.ipynb    # EDA, feature engineering, model training & evaluation notebook
├── app.py                                 # Main Streamlit web application & interactive UI
├── features.py                            # Modular pre-transaction feature engineering pipeline
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation

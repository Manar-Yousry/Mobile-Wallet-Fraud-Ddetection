# 🛡️ Mobile-Wallet Fraud Detection & Cost-Based Threshold Tuner

An end-to-end Machine Learning system and interactive **Streamlit** decision-support application for real-time mobile-wallet fraud detection. Beyond binary classification, this project incorporates **cost-sensitive threshold tuning**, allowing risk teams to dynamically balance the financial cost of false alarms against missed fraud cases.

---

## ✨ Key Features

- **Leakage-Free Feature Engineering (`features.py`):** Uses strictly pre-transaction data available at the moment of authorization (no post-transaction balances), preventing target leakage and ensuring production readiness.
- **Imbalanced Learning Pipeline:** Trained using a **Random Forest Classifier** with **10:1 Undersampling**, achieving a **Test PR-AUC of 1.0000 (95% CI: 1.000–1.000)** on synthetic mobile money transaction data (744 hourly time steps / 31 days).
- **Interactive Cost-Matrix & Threshold Tuner:**
  - Adjust the **Classification Threshold** in real time.
  - Customize **False Alarm Cost** (cost of blocking an honest user, e.g., `$15.00`) and **Missed Fraud Cost** (financial loss per undetected fraud, e.g., `$500.00`).
- **Dual Evaluation Modes (`app.py`):**
  1. **Single Transaction Scoring:** Evaluate individual transactions on the fly by inputting transaction type, amount, time step, and pre-transaction sender/recipient balances.
  2. **Batch CSV Upload & Cost Simulator:** Upload bulk transaction logs (`.csv`) to run batch predictions and simulate total financial impact across different decision thresholds.

---

## 📂 Repository Structure

```text
├── .devcontainer/                         # Dev container configuration for reproducible environments
├── artifacts/                             # Saved model binaries, scalers, and metadata artifacts
├── Mobile_Wallet_Fraud_Detection.ipynb    # Full EDA, feature engineering, model training & evaluation notebook
├── app.py                                 # Streamlit web application & cost-based threshold simulator
├── features.py                            # Reusable pre-transaction feature engineering pipeline
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import build_features

st.set_page_config(page_title="Mobile-Wallet Fraud Detection", page_icon="🛡️", layout="wide")

REQUIRED_COLS = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']

@st.cache_resource
def load_bundle():
    return joblib.load("artifacts/fraud_model_bundle.joblib")

def score(df_input: pd.DataFrame) -> np.ndarray:
    """Fraud probability per row. Only TRANSFER and CASH_OUT can be fraud; other types score 0."""
    scores = np.zeros(len(df_input))
    mask = df_input['type'].isin(['TRANSFER', 'CASH_OUT']).to_numpy()
    if mask.any():
        X = build_features(df_input.loc[mask])[feature_cols]
        scores[mask] = model.predict_proba(X)[:, 1]
    return scores

bundle = load_bundle()
model = bundle['model']
feature_cols = bundle['feature_cols']
opt_thresh = bundle['optimal_threshold']
ci = bundle.get('test_pr_auc_ci')
ci_text = f" (95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci else ""

st.title("🛡️ Mobile-Wallet Fraud Detection & Cost-Based Threshold Tuner")
st.markdown(f"**Model:** `{bundle['model_name']}` | **Test PR-AUC:** `{bundle['test_pr_auc']:.4f}`{ci_text} | **Chosen Threshold:** `{opt_thresh:.2f}`")
st.caption("The model uses only information available before a transaction is approved (no post-transaction balances).")

# Sidebar: Threshold Slider & Cost Matrix Parameters
st.sidebar.header("⚙️ Decision Threshold & Cost Matrix")
threshold = st.sidebar.slider(
    "Classification Threshold", min_value=0.01, max_value=0.99, value=float(opt_thresh), step=0.01,
    help="Transactions with fraud probability >= threshold are flagged/blocked."
)
cost_fp = st.sidebar.number_input("False Alarm Cost (Blocked Honest User $)", min_value=1.0, value=float(bundle['default_cost_fp']), step=5.0)
cost_fn = st.sidebar.number_input("Missed Fraud Cost ($ per missed fraud)", min_value=10.0, value=float(bundle['default_cost_fn']), step=50.0)

tab1, tab2 = st.tabs(["📄 Single Transaction Scoring", "📂 Batch CSV Upload & Cost Simulator"])

# --- TAB 1: Single Transaction Form ---
with tab1:
    st.subheader("Evaluate a Single Mobile-Wallet Transaction")
    with st.form("tx_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
            step = st.number_input("Time Step (Hour 1-744)", min_value=1, max_value=744, value=250)
        with col2:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150000.0, step=100.0)
            old_orig = st.number_input("Sender Balance Before ($)", min_value=0.0, value=150000.0, step=100.0)
        with col3:
            old_dest = st.number_input("Recipient Balance Before ($)", min_value=0.0, value=0.0, step=100.0)
        submitted = st.form_submit_button("Calculate Fraud Score")

    if submitted:
        row_df = pd.DataFrame([{'step': step, 'type': tx_type, 'amount': amount,
                                'oldbalanceOrg': old_orig, 'oldbalanceDest': old_dest}])
        fraud_prob = float(score(row_df)[0])
        is_flagged = fraud_prob >= threshold
        c1, c2 = st.columns(2)
        c1.metric("Fraud Probability Score", f"{fraud_prob:.4f}")
        c2.metric("Decision", "🚨 BLOCK / REVIEW (FRAUD)" if is_flagged else "✅ APPROVE (LEGITIMATE)")

# --- TAB 2: Batch CSV Upload & Threshold Cost Analysis ---
with tab2:
    st.subheader("Upload Transactions CSV for Batch Scoring & Cost Impact")
    uploaded_file = st.file_uploader("Upload PaySim-format CSV (or use artifacts/sample_transactions.csv)", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        missing = [c for c in REQUIRED_COLS if c not in raw_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            raw_df['fraud_score'] = np.round(score(raw_df), 5)
            raw_df['predicted_fraud'] = (raw_df['fraud_score'] >= threshold).astype(int)
            st.write(f"Scored **{len(raw_df):,}** transactions | Flagged as Fraud: **{raw_df['predicted_fraud'].sum():,}**")

            if 'isFraud' in raw_df.columns:
                y_true = raw_df['isFraud'].values
                y_pred = raw_df['predicted_fraud'].values
                tp = int(((y_pred == 1) & (y_true == 1)).sum())
                fn = int(((y_pred == 0) & (y_true == 1)).sum())
                fp = int(((y_pred == 1) & (y_true == 0)).sum())
                fp_loss, fn_loss = fp * cost_fp, fn * cost_fn
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fraud Caught (TP)", f"{tp:,}")
                m2.metric("Missed Fraud (FN)", f"{fn:,}", delta=f"-${fn_loss:,.0f} loss", delta_color="inverse")
                m3.metric("False Alarms (FP)", f"{fp:,}", delta=f"-${fp_loss:,.0f} cost", delta_color="inverse")
                m4.metric("Total Financial Loss", f"${fp_loss + fn_loss:,.2f}")

            show_cols = [c for c in ['type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'fraud_score', 'predicted_fraud'] if c in raw_df.columns]
            st.dataframe(raw_df[show_cols].head(50), use_container_width=True)
            st.download_button("⬇️ Download Scored Transactions CSV", data=raw_df.to_csv(index=False).encode('utf-8'),
                               file_name="scored_transactions.csv", mime="text/csv")

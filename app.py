import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mobile-Wallet Fraud Detection", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load("fraud_model_bundle.joblib")

def prepare_features(df_input: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df_feat = df_input.copy()
    df_feat['is_transfer'] = (df_feat['type'] == 'TRANSFER').astype('int8')
    df_feat['orig_balance_diff'] = df_feat['oldbalanceOrg'] - df_feat['newbalanceOrig']
    df_feat['dest_balance_diff'] = df_feat['newbalanceDest'] - df_feat['oldbalanceDest']
    df_feat['errorBalanceOrig'] = df_feat['newbalanceOrig'] + df_feat['amount'] - df_feat['oldbalanceOrg']
    df_feat['errorBalanceDest'] = df_feat['oldbalanceDest'] + df_feat['amount'] - df_feat['newbalanceDest']
    df_feat['amount_to_oldOrig_ratio'] = df_feat['amount'] / (df_feat['oldbalanceOrg'] + 1e-5)
    df_feat['amount_to_oldDest_ratio'] = df_feat['amount'] / (df_feat['oldbalanceDest'] + 1e-5)
    df_feat['is_orig_drained'] = ((df_feat['oldbalanceOrg'] > 0) & (df_feat['newbalanceOrig'] == 0)).astype('int8')
    df_feat['is_dest_zero_all'] = ((df_feat['oldbalanceDest'] == 0) & (df_feat['newbalanceDest'] == 0) & (df_feat['amount'] > 0)).astype('int8')
    df_feat['is_orig_zero_all'] = ((df_feat['oldbalanceOrg'] == 0) & (df_feat['newbalanceOrig'] == 0)).astype('int8')
    df_feat['exact_balance_transfer'] = (np.isclose(df_feat['amount'], df_feat['oldbalanceOrg'], atol=1e-2)).astype('int8')
    df_feat['hour_of_day'] = (df_feat['step'] % 24).astype('int8')
    return df_feat[feature_cols]

bundle = load_bundle()
model = bundle['model']
feature_cols = bundle['feature_cols']
opt_thresh = bundle['optimal_threshold']

st.title("🛡️ Mobile-Wallet Fraud Detection & Cost-Based Threshold Tuner")
st.markdown(f"**Model:** `{bundle['model_name']}` | **Test PR-AUC:** `{bundle['test_pr_auc']:.4f}` | **Cost-Optimal Threshold:** `{opt_thresh:.2f}`")

# Sidebar: Threshold Slider & Cost Matrix Parameters
st.sidebar.header("⚙️ Decision Threshold & Cost Matrix")
threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(opt_thresh),
    step=0.01,
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
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150000.0, step=100.0)
            step = st.number_input("Time Step (Hour 1-744)", min_value=1, max_value=744, value=250)
        with col2:
            old_orig = st.number_input("Sender Old Balance ($)", min_value=0.0, value=150000.0, step=100.0)
            new_orig = st.number_input("Sender New Balance ($)", min_value=0.0, value=0.0, step=100.0)
        with col3:
            old_dest = st.number_input("Recipient Old Balance ($)", min_value=0.0, value=0.0, step=100.0)
            new_dest = st.number_input("Recipient New Balance ($)", min_value=0.0, value=0.0, step=100.0)

        submitted = st.form_submit_button("Calculate Fraud Score")

    if submitted:
        if tx_type not in ["TRANSFER", "CASH_OUT"]:
            fraud_prob = 0.0
        else:
            row_df = pd.DataFrame([{
                'step': step, 'type': tx_type, 'amount': amount,
                'oldbalanceOrg': old_orig, 'newbalanceOrig': new_orig,
                'oldbalanceDest': old_dest, 'newbalanceDest': new_dest
            }])
            X_single = prepare_features(row_df, feature_cols)
            fraud_prob = float(model.predict_proba(X_single)[:, 1][0])

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
        scores = np.zeros(len(raw_df))
        eligible_mask = raw_df['type'].isin(['TRANSFER', 'CASH_OUT'])

        if eligible_mask.sum() > 0:
            X_batch = prepare_features(raw_df.loc[eligible_mask], feature_cols)
            scores[eligible_mask] = model.predict_proba(X_batch)[:, 1]

        raw_df['fraud_score'] = np.round(scores, 5)
        raw_df['predicted_fraud'] = (raw_df['fraud_score'] >= threshold).astype(int)

        st.write(f"Scored **{len(raw_df):,}** transactions | Flagged as Fraud: **{raw_df['predicted_fraud'].sum():,}**")

        # If ground-truth isFraud exists in uploaded CSV, compute Missed Fraud vs False Alarms & Cost
        if 'isFraud' in raw_df.columns:
            y_true = raw_df['isFraud'].values
            y_pred = raw_df['predicted_fraud'].values
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fp_loss = fp * cost_fp
            fn_loss = fn * cost_fn
            total_loss = fp_loss + fn_loss

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fraud Caught (TP)", f"{tp:,}")
            m2.metric("Missed Fraud (FN)", f"{fn:,}", delta=f"-${fn_loss:,.0f} loss", delta_color="inverse")
            m3.metric("False Alarms (FP)", f"{fp:,}", delta=f"-${fp_loss:,.0f} cost", delta_color="inverse")
            m4.metric("Total Financial Loss", f"${total_loss:,.2f}")

        st.dataframe(raw_df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'fraud_score', 'predicted_fraud']].head(50), use_container_width=True)
        st.download_button(
            "⬇️ Download Scored Transactions CSV",
            data=raw_df.to_csv(index=False).encode('utf-8'),
            file_name="scored_transactions.csv",
            mime="text/csv"
        )

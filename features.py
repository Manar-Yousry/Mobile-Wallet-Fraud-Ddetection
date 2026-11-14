"""Shared feature engineering for training (notebook) and serving (Streamlit app).

SAFE_FEATURES only use information that exists BEFORE a transaction is approved.
POST_TX_FEATURES use balances after the transaction. In PaySim they leak the label
and would not be available in a real-time decision, so they are NOT used by the model.
"""
import numpy as np
import pandas as pd

SAFE_FEATURES = [
    'is_transfer', 'amount', 'log_amount', 'oldbalanceOrg', 'oldbalanceDest',
    'amount_to_oldOrig_ratio', 'amount_to_oldDest_ratio',
    'amount_ge_old_orig', 'exact_balance_transfer',
    'is_orig_old_zero', 'is_dest_old_zero', 'hour_of_day',
]

POST_TX_FEATURES = [
    'newbalanceOrig', 'newbalanceDest', 'orig_balance_diff', 'dest_balance_diff',
    'errorBalanceOrig', 'errorBalanceDest', 'is_orig_drained', 'is_dest_zero_all',
]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adds model features. Post-transaction features are added only if those columns exist."""
    d = data.copy()
    # float64 avoids float32 rounding on large amounts (float32 keeps ~7 digits only)
    num_cols = [c for c in ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                            'oldbalanceDest', 'newbalanceDest'] if c in d.columns]
    for c in num_cols:
        d[c] = d[c].astype('float64')

    d['is_transfer'] = (d['type'] == 'TRANSFER').astype('int8')
    d['log_amount'] = np.log1p(d['amount'])
    d['amount_to_oldOrig_ratio'] = d['amount'] / (d['oldbalanceOrg'] + 1.0)
    d['amount_to_oldDest_ratio'] = d['amount'] / (d['oldbalanceDest'] + 1.0)
    d['amount_ge_old_orig'] = ((d['oldbalanceOrg'] > 0) & (d['amount'] >= d['oldbalanceOrg'] - 0.01)).astype('int8')
    d['exact_balance_transfer'] = np.isclose(d['amount'], d['oldbalanceOrg'], atol=1e-2).astype('int8')
    d['is_orig_old_zero'] = (d['oldbalanceOrg'] == 0).astype('int8')
    d['is_dest_old_zero'] = (d['oldbalanceDest'] == 0).astype('int8')
    d['hour_of_day'] = (d['step'] % 24).astype('int8')

    if {'newbalanceOrig', 'newbalanceDest'}.issubset(d.columns):
        d['orig_balance_diff'] = d['oldbalanceOrg'] - d['newbalanceOrig']
        d['dest_balance_diff'] = d['newbalanceDest'] - d['oldbalanceDest']
        d['errorBalanceOrig'] = d['newbalanceOrig'] + d['amount'] - d['oldbalanceOrg']
        d['errorBalanceDest'] = d['oldbalanceDest'] + d['amount'] - d['newbalanceDest']
        d['is_orig_drained'] = ((d['oldbalanceOrg'] > 0) & (d['newbalanceOrig'] == 0)).astype('int8')
        d['is_dest_zero_all'] = ((d['oldbalanceDest'] == 0) & (d['newbalanceDest'] == 0) & (d['amount'] > 0)).astype('int8')
    return d

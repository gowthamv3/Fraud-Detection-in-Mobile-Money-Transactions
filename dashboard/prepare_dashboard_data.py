"""Creates the CSV files used by the Power BI dashboard. Run from the dashboard folder:
    python prepare_dashboard_data.py
Needs ../Fraud_Analysis_Dataset.csv, ../fraud_model_v1.pkl and ../validation_results.json."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(OUT, exist_ok=True)
THRESHOLD = 0.5   # recommended threshold, see the project documentation
TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']


def build_features(data):
    X = data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].astype(float)
    X['errorBalanceOrig'] = X.oldbalanceOrg - X.amount - X.newbalanceOrig
    X['errorBalanceDest'] = X.oldbalanceDest + X.amount - X.newbalanceDest
    X['orig_emptied'] = (X.newbalanceOrig == 0).astype(int)
    X['amount_to_balance'] = X.amount / (X.oldbalanceOrg + 1)
    X['log_amount'] = np.log1p(X.amount)
    X['dest_is_merchant'] = data.nameDest.str.startswith('M').astype(int)
    for t in TYPES:
        X[f'type_{t}'] = (data.type == t).astype(int)
    return X


df = pd.read_csv(os.path.join(ROOT, 'Fraud_Analysis_Dataset.csv'))
X = build_features(df)
art = joblib.load(os.path.join(ROOT, 'fraud_model_v1.pkl'))

# same split as the notebook
train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df.isFraud, random_state=42)

t = pd.DataFrame({'TxnID': df.index + 1, 'Hour': df.step, 'Type': df.type, 'Amount': df.amount.round(2),
                  'SenderBalanceBefore': df.oldbalanceOrg, 'SenderBalanceAfter': df.newbalanceOrig,
                  'ReceiverBalanceBefore': df.oldbalanceDest, 'ReceiverBalanceAfter': df.newbalanceDest})
t['ReceiverKind'] = np.where(df.nameDest.str.startswith('M'), 'Merchant', 'Customer')
t['IsFraud'] = df.isFraud
t['Class'] = np.where(df.isFraud == 1, 'Fraud', 'Normal')
t['AccountEmptied'] = np.where(df.newbalanceOrig == 0, 'Yes', 'No')
t['WholeBalanceMoved'] = np.where((df.amount - df.oldbalanceOrg).abs() < 1, 'Yes', 'No')
t['AmountBand'] = pd.cut(df.amount, [0, 1e3, 1e4, 1e5, 1e6, 1e12],
                         labels=['1. Under 1K', '2. 1K-10K', '3. 10K-100K', '4. 100K-1M', '5. Over 1M']).astype(str)
t['HourPeriod'] = np.where(df.step <= 7, 'Hours 1-7', 'Hours 8-95')
t['Split'] = 'Train'
t.loc[test_idx, 'Split'] = 'Test'
t['FraudProbability'] = art['model'].predict_proba(X[art['features']])[:, 1].round(5)
pred = (t.FraudProbability >= THRESHOLD).astype(int)
t['Predicted'] = np.where(pred == 1, 'Fraud', 'Normal')
t['Outcome'] = np.select([(pred == 1) & (t.IsFraud == 1), (pred == 1) & (t.IsFraud == 0), (pred == 0) & (t.IsFraud == 1)],
                         ['Fraud caught', 'False alarm', 'Missed fraud'], 'Correct normal')
t.to_csv(os.path.join(OUT, 'transactions.csv'), index=False)

v = json.load(open(os.path.join(ROOT, 'validation_results.json')))


def exact(m):
    """Recompute from counts, so values are not rounded twice."""
    tp, fp, fn = m['tp'], m['fp'], m['fn']
    p, r = tp / (tp + fp), tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)


rows = []
for name, m in v['cv_model_selection'].items():
    p, r, f = exact(m)
    rows.append({'Model': name, 'Precision': p, 'Recall': r, 'F1': f, 'PR_AUC': m['pr_auc'],
                 'FalseAlarms': m['fp'], 'MissedFraud': m['fn']})
pd.DataFrame(rows).sort_values('F1', ascending=False).to_csv(os.path.join(OUT, 'model_comparison.csv'), index=False)

imp = pd.Series(art['model'].feature_importances_, index=art['features']).sort_values(ascending=False)
pd.DataFrame({'Feature': imp.index, 'Importance': imp.values.round(4)}).head(8).to_csv(os.path.join(OUT, 'feature_importance.csv'), index=False)

fs = v['feature_sets']
checks = [('1. All 16 features', 'Test set', fs['A post-transaction (16 features)']['test']),
          ('2. Pre-authorization features', 'Test set', fs['B pre-authorization']['test']),
          ('3. No balance features', 'Test set', fs['C no balances']['test']),
          ('4. Hours 1-7 only', '5-fold CV, 78 fraud', v['same_window_steps_1_7']['A']),
          ('5. Train hours 1-5, test 6-7', 'Time split, 34 fraud', v['chronological']['A'])]
pd.DataFrame([dict(zip(('Check', 'Evaluation', 'Precision', 'Recall', 'F1'), (c, e) + exact(m)))
              for c, e, m in checks]).to_csv(os.path.join(OUT, 'validation_checks.csv'), index=False)

test = t[t.Split == 'Test']
print(len(t), 'rows; test outcomes:', test.Outcome.value_counts().to_dict())
print('fraud share of money:', round(t[t.IsFraud == 1].Amount.sum() / t.Amount.sum(), 4))

"""Extra validation for Version 1. Run from this folder:  python validation_checks.py
Writes validation_results.json. Does not change the notebook or the saved model."""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
SEED = 42
TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
BEST = dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=1.0,
            min_child_weight=3, eval_metric='logloss', random_state=SEED, n_jobs=-1)


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


TYPE_COLS = [f'type_{t}' for t in TYPES]
# Model A: everything, including balances recorded after the transaction
MODEL_A = None
# Model B: only what is known before the transaction is approved
MODEL_B = ['amount', 'log_amount', 'oldbalanceOrg', 'oldbalanceDest', 'amount_to_balance', 'dest_is_merchant'] + TYPE_COLS
# Model C: pre-authorization without any balance at all
MODEL_C = ['amount', 'log_amount', 'dest_is_merchant'] + TYPE_COLS


def scores(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'precision': round(precision_score(y, pred, zero_division=0), 4), 'recall': round(recall_score(y, pred), 4),
            'f1': round(f1_score(y, pred), 4), 'roc_auc': round(roc_auc_score(y, proba), 4),
            'pr_auc': round(average_precision_score(y, proba), 4), 'fpr': round(fp / (fp + tn), 5),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def xgb(y):
    return XGBClassifier(**BEST, scale_pos_weight=(y == 0).sum() / (y == 1).sum())


df = pd.read_csv('Fraud_Analysis_Dataset.csv')
X, y = build_features(df), df.isFraud
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
out = {}

# 1. Model selection using only the training set
spw = (ytr == 0).sum() / (ytr == 1).sum()
baselines = {
    'Logistic Regression': Pipeline([('s', StandardScaler()), ('c', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED))]),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=6, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=SEED),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, scale_pos_weight=spw, eval_metric='logloss', random_state=SEED, n_jobs=-1),
}
out['cv_model_selection'] = {n: scores(ytr, cross_val_predict(m, Xtr, ytr, cv=cv, method='predict_proba')[:, 1])
                             for n, m in baselines.items()}

# 2. Final saved model on the test set, at both thresholds
art = joblib.load('fraud_model_v1.pkl')
proba = art['model'].predict_proba(Xte[art['features']])[:, 1]
out['saved_model_test'] = {'t=0.50': scores(yte, proba, 0.5), f"t={art['threshold']:.3f}": scores(yte, proba, art['threshold'])}

# 3. Post-transaction vs pre-authorization feature sets
out['feature_sets'] = {}
for name, cols in [('A post-transaction (16 features)', list(X.columns)), ('B pre-authorization', MODEL_B), ('C no balances', MODEL_C)]:
    m = xgb(ytr).fit(Xtr[cols], ytr)
    out['feature_sets'][name] = {
        'cv_train': scores(ytr, cross_val_predict(xgb(ytr), Xtr[cols], ytr, cv=cv, method='predict_proba')[:, 1]),
        'test': scores(yte, m.predict_proba(Xte[cols])[:, 1])}

# 4. Only TRANSFER and CASH_OUT, the types where fraud exists
mask = df.type.isin(['TRANSFER', 'CASH_OUT'])
out['transfer_cashout_only'] = {'rows': int(mask.sum()), 'fraud': int(y[mask].sum())}
for name, cols in [('A', list(X.columns)), ('B', MODEL_B)]:
    out['transfer_cashout_only'][name] = scores(y[mask], cross_val_predict(xgb(y[mask]), X[mask][cols], y[mask], cv=cv, method='predict_proba')[:, 1])

# 5. Same time window: steps 1-7 only, so normal and fraud come from the same hours
w = df.step <= 7
out['same_window_steps_1_7'] = {'rows': int(w.sum()), 'fraud': int(y[w].sum())}
for name, cols in [('A', list(X.columns)), ('B', MODEL_B)]:
    out['same_window_steps_1_7'][name] = scores(y[w], cross_val_predict(xgb(y[w]), X[w][cols], y[w], cv=cv, method='predict_proba')[:, 1])

# 6. Chronological holdout inside the window: train on steps 1-5, test on steps 6-7
tr, te = df.step <= 5, (df.step >= 6) & (df.step <= 7)
out['chronological'] = {'train_rows': int(tr.sum()), 'train_fraud': int(y[tr].sum()), 'test_rows': int(te.sum()), 'test_fraud': int(y[te].sum())}
for name, cols in [('A', list(X.columns)), ('B', MODEL_B)]:
    m = xgb(y[tr]).fit(X[tr][cols], y[tr])
    out['chronological'][name] = scores(y[te], m.predict_proba(X[te][cols])[:, 1])

# 7. Are TRANSFER and CASH_OUT frauds really linked in the data?
f = df[df.isFraud == 1]
t, c = f[f.type == 'TRANSFER'], f[f.type == 'CASH_OUT']
out['transfer_cashout_link'] = {
    'transfers': len(t), 'cashouts': len(c),
    'transfer_amount_also_in_cashout': round(float(t.amount.isin(c.amount).mean()), 4),
    'transfer_receiver_is_cashout_sender': round(float(t.nameDest.isin(c.nameOrig).mean()), 4)}

# 8. Dataset facts
out['dataset'] = {'rows': len(df), 'fraud': int(y.sum()), 'unique_steps_fraud': int(df[y == 1].step.nunique()),
                  'unique_steps_normal': int(df[y == 0].step.nunique()), 'fraud_steps_1_7': int(y[w].sum())}

json.dump(out, open('validation_results.json', 'w'), indent=2)
print(json.dumps(out, indent=1))

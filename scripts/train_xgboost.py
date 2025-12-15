# train_xgboost.py
# Trains XGBoost Classifier for 3-month and 6-month direction prediction

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

# ========================= CONFIG =========================
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

HORIZONS = ['3m', '6m']

# Load previous results
try:
    prev_results = pd.read_csv(f'{MODELS_DIR}/model_comparison_rf_vs_logistic.csv', index_col=0)
except:
    prev_results = None

results = {}

print("Starting XGBoost training for 3-month and 6-month horizons...\n")

for horizon in HORIZONS:
    print(f"=== Training XGBoost for {horizon.upper()} horizon ===")
    
    train = pd.read_csv(f'{PROCESSED_DIR}/train_{horizon}.csv')
    val   = pd.read_csv(f'{PROCESSED_DIR}/val_{horizon}.csv')
    test  = pd.read_csv(f'{PROCESSED_DIR}/test_{horizon}.csv')
    
    with open(f'{PROCESSED_DIR}/features_{horizon}.txt', 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    target = f'target_up_{horizon}'
    
    X_train = train[features]
    y_train = train[target]
    X_val = val[features]
    y_val = val[target]
    X_test = test[features]
    y_test = test[target]
    
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    # Scale_pos_weight for imbalance
    neg, pos = np.bincount(y_train_val)
    scale_pos_weight = neg / pos
    
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [6, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [scale_pos_weight]
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_val, y_train_val)
    
    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob)
    }
    
    results[horizon] = metrics
    
    print(f"   Test Performance ({horizon}):")
    for k, v in metrics.items():
        print(f"     {k}: {v:.4f}")
    
    # Feature importance
    importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n   Top 10 features ({horizon}):")
    print(importances.head(10).to_string())
    print()
    
    joblib.dump(best_model, f'{MODELS_DIR}/xgboost_{horizon}.pkl')
    importances.to_csv(f'{MODELS_DIR}/xgb_feature_importance_{horizon}.csv')

# Comparison
print("=== FINAL XGBoost RESULTS ===")
xgb_df = pd.DataFrame(results).T.round(4)
print(xgb_df)

if prev_results is not None:
    full_comparison = pd.concat([prev_results, xgb_df.add_suffix(' (XGB)')], axis=1)
    print("\nFull model comparison:")
    print(full_comparison)
    full_comparison.to_csv(f'{MODELS_DIR}/full_model_comparison.csv')

print(f"\nXGBoost models saved in {MODELS_DIR}/")
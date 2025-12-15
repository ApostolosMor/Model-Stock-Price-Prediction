# train_lightgbm.py
# Trains LightGBM Classifier for 3-month and 6-month direction prediction

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

# ========================= CONFIG =========================
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

HORIZONS = ['3m', '6m']

# Load previous comparison if exists
try:
    prev_comparison = pd.read_csv(f'{MODELS_DIR}/full_model_comparison.csv', index_col=0)
except:
    prev_comparison = None

results = {}

print("Starting LightGBM training for 3-month and 6-month horizons...\n")

for horizon in HORIZONS:
    print(f"=== Training LightGBM for {horizon.upper()} horizon ===")
    
    # Load data
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
    
    # Combine train + val for fitting
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    # Calculate scale_pos_weight for class imbalance
    neg, pos = np.bincount(y_train_val)
    scale_pos_weight = neg / pos if pos > 0 else 1
    
    # LightGBM dataset
    lgb_train = lgb.Dataset(X_train_val, label=y_train_val)
    
    # Parameter grid
    param_grid = {
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [200, 400],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [scale_pos_weight]
    }
    
    # Base model
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    # Grid search
    grid_search = GridSearchCV(
        lgb_model, 
        param_grid, 
        cv=3, 
        scoring='roc_auc', 
        n_jobs=-1
    )
    grid_search.fit(X_train_val, y_train_val)
    
    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    
    # Predict on test
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
    
    # Save
    joblib.dump(best_model, f'{MODELS_DIR}/lightgbm_{horizon}.pkl')
    importances.to_csv(f'{MODELS_DIR}/lgb_feature_importance_{horizon}.csv')

# Results table
print("=== LightGBM RESULTS ===")
lgb_df = pd.DataFrame(results).T.round(4)
print(lgb_df)

# Full comparison
if prev_comparison is not None:
    full_comp = pd.concat([prev_comparison, lgb_df.add_suffix(' (LGB)')], axis=1)
    print("\nFull model comparison:")
    print(full_comp)
    full_comp.to_csv(f'{MODELS_DIR}/full_model_comparison_with_lgb.csv')
else:
    lgb_df.to_csv(f'{MODELS_DIR}/lightgbm_results.csv')

print(f"\nLightGBM models and results saved in {MODELS_DIR}/")
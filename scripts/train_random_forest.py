# train_random_forest.py
# Trains Random Forest Classifier for 3-month and 6-month direction prediction

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

# ========================= CONFIG =========================
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

HORIZONS = ['3m', '6m']

# Load previous Logistic results for comparison
try:
    logistic_results = pd.read_csv(f'{MODELS_DIR}/logistic_regression_results.csv', index_col=0)
except:
    logistic_results = None

results = {}

print("Starting Random Forest training for 3-month and 6-month horizons...\n")

for horizon in HORIZONS:
    print(f"=== Training Random Forest for {horizon.upper()} horizon ===")
    
    # Load data
    train = pd.read_csv(f'{PROCESSED_DIR}/train_{horizon}.csv', parse_dates=['Date'])
    val   = pd.read_csv(f'{PROCESSED_DIR}/val_{horizon}.csv', parse_dates=['Date'])
    test  = pd.read_csv(f'{PROCESSED_DIR}/test_{horizon}.csv', parse_dates=['Date'])
    
    with open(f'{PROCESSED_DIR}/features_{horizon}.txt', 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    target = f'target_up_{horizon}'
    
    X_train = train[features]
    y_train = train[target]
    X_val = val[features]
    y_val = val[target]
    X_test = test[features]
    y_test = test[target]
    
    # Simple hyperparameter tuning
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    
    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    
    # Final prediction on test
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
    print()
    
    # Feature importance
    importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"   Top 10 important features ({horizon}):")
    print(importances.head(10).to_string())
    print()
    
    # Save model and importances
    joblib.dump(best_model, f'{MODELS_DIR}/random_forest_{horizon}.pkl')
    importances.to_csv(f'{MODELS_DIR}/rf_feature_importance_{horizon}.csv')

# Final comparison
print("=== RESULTS COMPARISON ===")
rf_df = pd.DataFrame(results).T.round(4)
if logistic_results is not None:
    comparison = pd.concat([logistic_results, rf_df], axis=1, keys=['Logistic', 'Random Forest'])
    print(comparison)
    comparison.to_csv(f'{MODELS_DIR}/model_comparison_rf_vs_logistic.csv')
else:
    print(rf_df)

print(f"\nAll done! Models and results saved in {MODELS_DIR}/")
# train_logistic_models.py
# Trains two Logistic Regression models: one for 3-month, one for 6-month ahead prediction

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.utils import compute_class_weight
import joblib
import os

# ========================= CONFIG =========================
PROCESSED_DIR = 'data/processed'

# Horizons we trained on
HORIZONS = ['3m', '6m']

# Model save path
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Results summary
results = {}

print("Starting Logistic Regression training for 3-month and 6-month horizons...\n")

for horizon in HORIZONS:
    print(f"=== Training Logistic Regression for {horizon.upper()} horizon ===")
    
    # Load data
    train = pd.read_csv(f'{PROCESSED_DIR}/train_{horizon}.csv', parse_dates=['Date'])
    val   = pd.read_csv(f'{PROCESSED_DIR}/val_{horizon}.csv', parse_dates=['Date'])
    test  = pd.read_csv(f'{PROCESSED_DIR}/test_{horizon}.csv', parse_dates=['Date'])
    
    # Load feature names
    with open(f'{PROCESSED_DIR}/features_{horizon}.txt', 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    target = f'target_up_{horizon}'
    
    # Prepare X and y
    X_train = train[features]
    y_train = train[target]
    
    X_val = val[features]
    y_val = val[target]
    
    X_test = test[features]
    y_test = test[target]
    
    # Combine train + val for final model (best practice after tuning)
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # Compute class weights to handle potential imbalance (common in stock direction)
    classes = np.unique(y_train_full)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_full)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"   Class distribution (train+val): {y_train_full.value_counts().to_dict()}")
    print(f"   Class weights: {class_weight_dict}")
    
    # Train Logistic Regression with balanced weights and L2 regularization
    model = LogisticRegression(
        penalty='l2',
        C=1.0,                  # inverse of regularization strength (can tune later)
        solver='lbfgs',
        max_iter=1000,
        class_weight=class_weight_dict,
        random_state=42
    )
    
    model.fit(X_train_full, y_train_full)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1
    
    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob)
    }
    
    results[horizon] = metrics
    
    print(f"   Test Performance ({horizon}):")
    for metric, value in metrics.items():
        print(f"     {metric}: {value:.4f}")
    print()
    
    # Save model
    model_path = f'{MODELS_DIR}/logistic_regression_{horizon}.pkl'
    joblib.dump(model, model_path)
    print(f"   Model saved to {model_path}\n")

# Final summary table
print("=== FINAL RESULTS SUMMARY ===")
summary_df = pd.DataFrame(results).T
summary_df = summary_df[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']]
print(summary_df.round(4))

# Save summary
summary_df.to_csv(f'{MODELS_DIR}/logistic_regression_results.csv')
print(f"\nResults saved to {MODELS_DIR}/logistic_regression_results.csv")
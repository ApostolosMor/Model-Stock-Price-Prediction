# train_stacking.py
# Trains proper stacking ensemble (LightGBM meta-learner) using only non-LSTM models

import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb

# ========================= CONFIG =========================
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
STACKED_DIR = 'models/stacked'
os.makedirs(STACKED_DIR, exist_ok=True)

HORIZONS = ['3m', '6m']

# Only non-LSTM models (avoid sequence length issues)
BASE_MODELS = {
    'Logistic': 'logistic_regression_{horizon}.pkl',
    'Random Forest': 'random_forest_{horizon}.pkl',
    'XGBoost': 'xgboost_{horizon}.pkl',
    'LightGBM': 'lightgbm_{horizon}.pkl',
}

print("Training stacking ensembles with LightGBM meta-learner (LSTM excluded for reliability)...\n")

for horizon in HORIZONS:
    print(f"--- {horizon.upper()} horizon ---")
    
    val = pd.read_csv(f'{PROCESSED_DIR}/val_{horizon}.csv', parse_dates=['Date'])
    with open(f'{PROCESSED_DIR}/features_{horizon}.txt', 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    target = f'target_up_{horizon}'
    
    val_probs = pd.DataFrame(index=val.index)
    used_models = []
    
    for model_name, filename_template in BASE_MODELS.items():
        filename = filename_template.format(horizon=horizon)
        path = os.path.join(MODELS_DIR, filename)
        
        if not os.path.exists(path):
            print(f"  {model_name} not found — skipping")
            continue
        
        model = joblib.load(path)
        probs = model.predict_proba(val[features])[:, 1]
        val_probs[model_name] = probs
        
        used_models.append(model_name)
    
    if len(used_models) < 2:
        print("Not enough models — skipping stacking")
        continue
    
    # No NaN expected — but drop just in case
    val_probs = val_probs.dropna()
    y_val = val[target].loc[val_probs.index]
    
    if len(val_probs) < 100:
        print("Too few rows — skipping")
        continue
    
    # Meta-learner: LightGBM
    meta_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=200
    )
    
    meta_model.fit(val_probs.values, y_val)
    
    # Save
    joblib.dump(meta_model, f'{STACKED_DIR}/stacked_meta_{horizon}.pkl')
    joblib.dump(used_models, f'{STACKED_DIR}/stacked_models_{horizon}.pkl')
    
    print(f"  Stacked successfully with: {used_models}")
    print(f"  Validation rows used: {len(val_probs)}\n")

print("Stacking training complete!")
print("Use predict_stacking.py for predictions with the stacked model.")
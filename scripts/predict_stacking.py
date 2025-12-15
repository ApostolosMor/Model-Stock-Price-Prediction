# predict_stacking.py
# Interactive prediction using trained stacking ensemble

import pandas as pd
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model

# ========================= CONFIG =========================
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
STACKED_DIR = 'models/stacked'
RAW_PRICES_PATH = 'data/raw/raw_stock_prices.csv'

HORIZONS = ['3m', '6m']
SEQUENCE_LENGTH = 12

TICKER = input("Enter ticker symbol (e.g., JPM): ").upper().strip()

if not TICKER:
    print("No ticker entered — exiting.")
    exit()

print(f"\nStacked ensemble predictions for {TICKER}\n")

prices = pd.read_csv(RAW_PRICES_PATH, sep=';', decimal=',', parse_dates=['Date'])
last_date = prices['Date'].max()
print(f"Last available data: {last_date.date()}\n")

for horizon in HORIZONS:
    print(f"--- {horizon.upper()} horizon ---")
    
    test = pd.read_csv(f'{PROCESSED_DIR}/test_{horizon}.csv', parse_dates=['Date'])
    with open(f'{PROCESSED_DIR}/features_{horizon}.txt', 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    ticker_data = test[test['Ticker'] == TICKER].sort_values('Date').tail(1)
    
    if ticker_data.empty:
        print("No data — skipping")
        continue
    
    current_price = ticker_data['price_close'].iloc[0]
    pred_date = ticker_data['Date'].iloc[0].date()
    print(f"Price: {current_price:.2f} | Date: {pred_date}")
    
    meta_path = f'{STACKED_DIR}/stacked_meta_{horizon}.pkl'
    models_path = f'{STACKED_DIR}/stacked_models_{horizon}.pkl'
    
    if not os.path.exists(meta_path):
        print("Stacked model not trained")
        continue
    
    meta_model = joblib.load(meta_path)
    base_models = joblib.load(models_path)
    
    base_probs = []
    
    for model_name in base_models:
        filename = {
            'Logistic': 'logistic_regression_{horizon}.pkl',
            'Random Forest': 'random_forest_{horizon}.pkl',
            'XGBoost': 'xgboost_{horizon}.pkl',
            'LightGBM': 'lightgbm_{horizon}.pkl',
            'LSTM': 'lstm_{horizon}.h5',
        }[model_name].format(horizon=horizon)
        
        path = os.path.join(MODELS_DIR, filename)
        model = joblib.load(path) if model_name != 'LSTM' else load_model(path)
        
        if model_name == 'LSTM':
            scaler = joblib.load(os.path.join(MODELS_DIR, f'lstm_scaler_{horizon}.pkl'))
            X_scaled = scaler.transform(ticker_data[features])
            X_seq = np.repeat(X_scaled, SEQUENCE_LENGTH, axis=0).reshape(1, SEQUENCE_LENGTH, len(features))
            prob = float(model.predict(X_seq, verbose=0)[0][0])
        else:
            prob = float(model.predict_proba(ticker_data[features])[0][1])
        
        base_probs.append(prob)
    
    stack_prob = meta_model.predict_proba(np.array(base_probs).reshape(1, -1))[0][1]
    stack_pred = "UP" if stack_prob > 0.5 else "DOWN"
    
    print(f"STACKED PREDICTION: {stack_pred} (probability {stack_prob:.4f})")
    print(f"Using {len(base_models)} base models: {', '.join(base_models)}\n")

print("Done!")
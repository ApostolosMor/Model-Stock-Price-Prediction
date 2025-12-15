# predict_ticker_interactive.py
# Interactive prediction with BLENDING ONLY (average of all models including LSTM)

import pandas as pd
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model

# ========================= CONFIG =========================
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
RAW_PRICES_PATH = 'data/raw/raw_stock_prices.csv'

HORIZONS = ['3m', '6m']
SEQUENCE_LENGTH = 12

BASE_MODELS = {
    'Logistic': 'logistic_regression_{horizon}.pkl',
    'Random Forest': 'random_forest_{horizon}.pkl',
    'XGBoost': 'xgboost_{horizon}.pkl',
    'LightGBM': 'lightgbm_{horizon}.pkl',
    'LSTM': 'lstm_{horizon}.h5',
}

TICKER = input("Enter ticker symbol (e.g., JPM): ").upper().strip()

if not TICKER:
    print("No ticker entered — exiting.")
    exit()

print(f"\nBlending ensemble predictions for {TICKER} from last available data...\n")

prices = pd.read_csv(RAW_PRICES_PATH, sep=';', decimal=',', parse_dates=['Date'])
last_date = prices['Date'].max()
print(f"Last available monthly data: {last_date.date()}\n")

for horizon in HORIZONS:
    print(f"--- Predictions for {horizon.upper()} horizon ---")
    
    test = pd.read_csv(f'{PROCESSED_DIR}/test_{horizon}.csv', parse_dates=['Date'])
    with open(f'{PROCESSED_DIR}/features_{horizon}.txt', 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    ticker_data = test[test['Ticker'] == TICKER].sort_values('Date').tail(1)
    
    if ticker_data.empty:
        print(f"No data for {TICKER} in {horizon} test set — skipping.")
        print()
        continue
    
    current_price = ticker_data['price_close'].iloc[0]
    pred_date = ticker_data['Date'].iloc[0].date()
    
    print(f"Current price (from data): {current_price:.2f}")
    print(f"Date used for prediction: {pred_date}\n")
    
    probs = {}
    available_models = []
    
    for model_name, filename_template in BASE_MODELS.items():
        filename = filename_template.format(horizon=horizon)
        path = os.path.join(MODELS_DIR, filename)
        
        if not os.path.exists(path):
            print(f"{model_name} model not found for {horizon}")
            continue
        
        if model_name == 'LSTM':
            scaler_path = os.path.join(MODELS_DIR, f'lstm_scaler_{horizon}.pkl')
            if not os.path.exists(scaler_path):
                print("LSTM scaler missing")
                continue
            scaler = joblib.load(scaler_path)
            model = load_model(path)
            
            X_scaled = scaler.transform(ticker_data[features])
            X_seq = np.repeat(X_scaled, SEQUENCE_LENGTH, axis=0)
            X_seq = X_seq.reshape(1, SEQUENCE_LENGTH, len(features))
            
            prob_up = float(model.predict(X_seq, verbose=0)[0][0])
        else:
            model = joblib.load(path)
            prob_up = float(model.predict_proba(ticker_data[features])[0][1])
        
        probs[model_name] = prob_up
        available_models.append(model_name)
    
    if not available_models:
        print("No models available for this horizon.")
        print()
        continue
    
    # Individual predictions
    print(f"{'Model':<15} {'Prob UP'}")
    print("-" * 25)
    for name, p in probs.items():
        print(f"{name:<15} {p:.4f}")
    print()
    
    # Blending ensemble
    blend_prob = np.mean(list(probs.values()))
    blend_pred = "UP" if blend_prob > 0.5 else "DOWN"
    
    print("BLENDING ENSEMBLE RESULT:")
    print(f"Prediction: {blend_pred}")
    print(f"Average Probability (UP): {blend_prob:.4f}")
    print(f"Based on {len(available_models)} models: {', '.join(available_models)}")
    print()

print("\nBlending predictions complete!")
# train_lstm.py
# Trains LSTM Classifier for 3-month and 6-month direction prediction
# Uses sequences of past 12 months per stock

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ========================= CONFIG =========================
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

HORIZONS = ['3m', '6m']
SEQUENCE_LENGTH = 12  # Use past 12 months to predict next

results = {}

print("Starting LSTM training for 3-month and 6-month horizons...\n")

def create_sequences(data, features, target, seq_length):
    X, y = [], []
    data_sorted = data.sort_values(['Ticker', 'Date'])
    for ticker, group in data_sorted.groupby('Ticker'):
        values = group[features].values
        targets = group[target].values
        if len(group) > seq_length:
            for i in range(len(group) - seq_length):
                X.append(values[i:i+seq_length])
                y.append(targets[i+seq_length])
    return np.array(X), np.array(y)

for horizon in HORIZONS:
    print(f"=== Training LSTM for {horizon.upper()} horizon ===")
    
    # Load data
    train = pd.read_csv(f'{PROCESSED_DIR}/train_{horizon}.csv', parse_dates=['Date'])
    val   = pd.read_csv(f'{PROCESSED_DIR}/val_{horizon}.csv', parse_dates=['Date'])
    test  = pd.read_csv(f'{PROCESSED_DIR}/test_{horizon}.csv', parse_dates=['Date'])
    
    with open(f'{PROCESSED_DIR}/features_{horizon}.txt', 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    
    target = f'target_up_{horizon}'
    
    # Combine train+val for fitting scaler
    full_train_val = pd.concat([train, val])
    
    # Scale features (fit on train+val)
    scaler = MinMaxScaler()
    scaler.fit(full_train_val[features])
    
    # Scale all sets
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    train_scaled[features] = scaler.transform(train[features])
    val_scaled[features] = scaler.transform(val[features])
    test_scaled[features] = scaler.transform(test[features])
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, features, target, SEQUENCE_LENGTH)
    X_val, y_val     = create_sequences(val_scaled, features, target, SEQUENCE_LENGTH)
    X_test, y_test   = create_sequences(test_scaled, features, target, SEQUENCE_LENGTH)
    
    print(f"   Sequences created: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    # LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Predict on test
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    
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
    
    # Save model and scaler
    model.save(f'{MODELS_DIR}/lstm_{horizon}.h5')
    joblib.dump(scaler, f'{MODELS_DIR}/lstm_scaler_{horizon}.pkl')

# Summary
print("=== LSTM RESULTS ===")
lstm_df = pd.DataFrame(results).T.round(4)
print(lstm_df)

print(f"\nLSTM models saved in {MODELS_DIR}/")
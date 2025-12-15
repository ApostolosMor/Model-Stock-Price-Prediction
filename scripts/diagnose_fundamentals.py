import pandas as pd

val = pd.read_csv('data/processed/val_3m.csv')
print("Val rows:", len(val))
print("Val NaNs in target:", val['target_up_3m'].isna().sum())

# Simulate LSTM length
seq_len = 12
lstm_len = len(val) - seq_len + 1
print("Expected LSTM predictions:", lstm_len)
print("Difference:", len(val) - lstm_len)
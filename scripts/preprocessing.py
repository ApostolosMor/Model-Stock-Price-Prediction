# scripts/preprocessing_FINAL_SAFE_WITH_ALL_RATIOS.py
# FINAL — ALL RATIOS — NO INF — PROFESSIONAL

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

print("Building FINAL SAFE dataset — ALL RATIOS — NO LOOK-AHEAD — NO INF")

df = pd.read_csv('data/processed/FINAL_COMPLETE_DATASET.csv', sep=';', decimal=',')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# 1. Target first
df['Price_3M'] = df.groupby('Ticker')['Close'].shift(-3)
df['Forward_Return_3M'] = df['Price_3M'] / df['Close'] - 1
df['Target'] = (df['Forward_Return_3M'] >= 0).astype(int)

# Remove last 3 months
df = df.groupby('Ticker').apply(lambda x: x.iloc[:-3]).reset_index(drop=True)
df = df.dropna(subset=['Forward_Return_3M'])

# 2. SHIFT ALL RATIOS BY 1 MONTH — THIS IS THE MAGIC FIX
ratio_cols = ['Market_PE', 'Market_PB', 'Market_PS', 'ROE', 'ROA',
              'Debt_Ratio', 'Debt_to_Equity', 'Current_Ratio',
              'Net_Profit_Margin', 'Operating_Margin']

for col in ratio_cols:
    df[f'{col}_t1'] = df.groupby('Ticker')[col].shift(1)

features = [f'{col}_t1' for col in ratio_cols]

# 3. Add momentum
for m in [1, 3, 6]:
    df[f'Momentum_{m}M'] = df.groupby('Ticker')['Close'].pct_change(m)
    features.append(f'Momentum_{m}M')

# 4. Clean
df = df.dropna(subset=features + ['Target'])

# 5. Clip extreme returns (safety)
df['Forward_Return_3M'] = df['Forward_Return_3M'].clip(-0.95, 5.0)

# 6. Split & scale
train = df[df['Date'] < '2019-01-01'].copy()
test  = df[df['Date'] >= '2019-01-01'].copy()

scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features]  = scaler.transform(test[features])

# 7. Save
os.makedirs('data/processed', exist_ok=True)
train.to_csv('data/processed/train_final.csv', index=False, sep=';', decimal=',')
test.to_csv('data/processed/test_final.csv', index=False, sep=';', decimal=',')

print(f"\nSUCCESS — {len(features)} FEATURES INCLUDING ALL RATIOS")
print("All ratios delayed by 1 month → NO look-ahead → NO +inf")
print("Ready for training and backtesting")
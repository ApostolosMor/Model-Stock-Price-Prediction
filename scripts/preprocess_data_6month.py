# preprocess_data_6month.py
# Creates train/val/test for 6-month-ahead stock direction prediction

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIG =========================
RAW_PRICES_PATH = 'data/raw/raw_stock_prices.csv'
RAW_FUNDS_PATH  = 'data/raw/raw_fundamentals_av.csv'
PROCESSED_DIR   = 'data/processed'

FORWARD_MONTHS = 6                     # ← 6-month prediction
TARGET_NAME = 'target_up_6m'            # ← target column name
# =========================================================

def load_prices():
    print("Loading prices (long format)...")
    df = pd.read_csv(RAW_PRICES_PATH, sep=';', decimal=',', parse_dates=['Date'])
    
    print(f"   Raw columns: {df.columns.tolist()}")
    print(f"   Raw shape: {df.shape}")
    
    # Safety: strip whitespace from column names (just in case)
    df.columns = df.columns.str.strip()
    
    # Rename the price column
    if 'Close' in df.columns:
        df = df.rename(columns={'Close': 'price_close'})
    elif 'close' in df.columns:
        df = df.rename(columns={'close': 'price_close'})
    else:
        raise ValueError(f"No price column found! Available: {df.columns.tolist()}")
    
    # Select and clean
    df = df[['Date', 'Ticker', 'price_close']].dropna(subset=['price_close'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    print(f"   Successfully loaded {len(df):,} price observations")
    print(f"   Unique tickers: {df['Ticker'].nunique()}")
    return df

def load_fundamentals():
    print("Loading fundamentals...")
    df = pd.read_csv(RAW_FUNDS_PATH, sep=';', decimal=',', parse_dates=['Date'], dayfirst=True)
    
    print(f"   Total rows before cleaning: {len(df)}")
    print(f"   Rows with invalid Date: {df['Date'].isna().sum()}")
    
    # CRITICAL FIX: drop the invalid date rows BEFORE any merging or indexing
    df = df.dropna(subset=['Date'])
    
    # Optional safety: keep only reasonable years
    df = df[df['Date'].dt.year >= 2000]
    
    # Strip whitespace from columns if needed
    df.columns = df.columns.str.strip()
    
    # Select only the columns we need (as before)
    cols = ['Date', 'Ticker', 'netIncome', 'totalRevenue', 'totalAssets', 'totalDebt',
            'totalShareholderEquity', 'currentAssets', 'currentLiabilities', 'operatingCashflow']
    df = df[[c for c in cols if c in df.columns]]
    
    # Convert to numeric
    for c in df.columns[2:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    print(f"   Final clean fundamental rows: {len(df)}")
    return df

def merge_asof(prices_df, funds_df):
    print("As-of merging fundamentals...")
    prices_df = prices_df.set_index('Date')
    funds_df  = funds_df.set_index('Date')
    merged_list = []
    for ticker in prices_df['Ticker'].unique():
        p = prices_df[prices_df['Ticker'] == ticker].copy()
        f = funds_df[funds_df['Ticker'] == ticker].copy()
        if f.empty or p.empty: continue
        f = f[~f.index.duplicated(keep='last')]
        merged = pd.merge_asof(p.sort_index(), f.sort_index(),
                               left_index=True, right_index=True,
                               by='Ticker', direction='backward',
                               tolerance=pd.Timedelta('120D'))
        merged_list.append(merged)
    return pd.concat(merged_list).reset_index()

def engineer_features_6m(df):
    print("Engineering features + 6-month target...")
    g = df.groupby('Ticker')
    
    # Momentum & volatility
    df['ret_1m']  = g['price_close'].pct_change(1)
    df['ret_3m']  = g['price_close'].pct_change(3)
    df['ret_6m']  = g['price_close'].pct_change(6)
    df['ret_12m'] = g['price_close'].pct_change(12)
    df['vol_6m']  = g['ret_1m'].rolling(6, min_periods=4).std().reset_index(0, drop=True)

    # Ratios
    df['pe'] = df['price_close'] * 1e6 / df['netIncome'].abs().replace(0, np.nan)
    df['pb'] = df['price_close'] * 1e6 / df['totalShareholderEquity'].replace(0, np.nan)
    df['de_ratio'] = df['totalDebt'] / df['totalShareholderEquity'].replace(0, np.nan)
    df['roa'] = df['netIncome'] / df['totalAssets'].replace(0, np.nan)
    df['roe'] = df['netIncome'] / df['totalShareholderEquity'].replace(0, np.nan)

    # Lags and growth
    for col in ['netIncome','totalRevenue','totalAssets','totalDebt',
                'totalShareholderEquity','operatingCashflow','pe','pb']:
        df[f'{col}_lag1'] = g[col].shift(1)
        df[f'{col}_growth'] = g[col].pct_change(1)

    # 6-MONTH FORWARD TARGET
    df['future_price'] = g['price_close'].shift(-FORWARD_MONTHS)
    df['future_return_6m'] = df['future_price'] / df['price_close'] - 1
    df[TARGET_NAME] = (df['future_return_6m'] > 0).astype(int)   # ← Fixed: was _3m

    df = df.dropna(subset=[TARGET_NAME])
    print(f"   → {len(df):,} rows with 6-month target")
    return df

def clean_winsorize_scale(df):
    feature_cols = [c for c in df.columns if c not in ['Date','Ticker',TARGET_NAME,
                                                       'future_price','future_return_6m']]  # ← Fixed name
    # Winsorize 1%/99%
    df[feature_cols] = df.groupby('Date')[feature_cols].transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
    # Imputation
    df[feature_cols] = df.groupby('Ticker')[feature_cols].ffill().fillna(df[feature_cols].median())
    # Standardize per month
    scaler = StandardScaler()
    df[feature_cols] = df.groupby('Date')[feature_cols].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten())
    return df, feature_cols

def split_and_save(df, features):
    train = df[df['Date'] < '2019-01-01']
    val   = df[(df['Date'] >= '2019-01-01') & (df['Date'] < '2022-01-01')]
    test  = df[df['Date'] >= '2022-01-01']
    
    train.to_csv(f'{PROCESSED_DIR}/train_6m.csv', index=False)
    val.to_csv(f'{PROCESSED_DIR}/val_6m.csv', index=False)
    test.to_csv(f'{PROCESSED_DIR}/test_6m.csv', index=False)      # ← Fixed typo: was PROCESSED}
    with open(f'{PROCESSED_DIR}/features_6m.txt', 'w') as f:
        f.write('\n'.join(features))

    print(f"\n6-month datasets saved in data/processed/")
    print(f"   train_6m.csv  → {len(train)} rows")
    print(f"   val_6m.csv    → {len(val)} rows")
    print(f"   test_6m.csv   → {len(test)} rows")

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = merge_asof(load_prices(), load_fundamentals())
    df = engineer_features_6m(df)           # ← Renamed function for clarity
    df, feats = clean_winsorize_scale(df)
    split_and_save(df, feats)

if __name__ == '__main__':
    main()
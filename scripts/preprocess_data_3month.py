# preprocess_data_3month.py
# With macro features + feature ranking for higher accuracy

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIG =========================
RAW_PRICES_PATH = 'data/raw/raw_stock_prices.csv'
RAW_FUNDS_PATH  = 'data/raw/raw_fundamentals_av.csv'
MACRO_PATH      = 'data/raw/macro_monthly.csv'  # VIX + 10Y yield
PROCESSED_DIR   = 'data/processed'

FORWARD_MONTHS = 3
TARGET_NAME = 'target_up_3m'

# =========================================================

def load_prices():
    print("Loading prices (long format)...")
    df = pd.read_csv(RAW_PRICES_PATH, sep=';', decimal=',', parse_dates=['Date'])
    
    df.columns = df.columns.str.strip()
    
    if 'Close' in df.columns:
        df = df.rename(columns={'Close': 'price_close'})
    else:
        raise ValueError(f"No price column found! Available: {df.columns.tolist()}")
    
    df = df[['Date', 'Ticker', 'price_close']].dropna(subset=['price_close'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    print(f"   Loaded {len(df):,} price observations, {df['Ticker'].nunique()} tickers")
    return df

def load_fundamentals():
    print("Loading fundamentals...")
    df = pd.read_csv(RAW_FUNDS_PATH, sep=';', decimal=',', parse_dates=['Date'], dayfirst=True)
    
    df = df.dropna(subset=['Date'])
    df = df[df['Date'].dt.year >= 2000]
    df.columns = df.columns.str.strip()
    
    cols = ['Date', 'Ticker', 'netIncome', 'totalRevenue', 'totalAssets', 'totalDebt',
            'totalShareholderEquity', 'currentAssets', 'currentLiabilities', 'operatingCashflow']
    df = df[[c for c in cols if c in df.columns]]
    
    for c in df.columns[2:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    print(f"   Clean fundamentals: {len(df)} rows")
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
                               by='Ticker', direction='backward', tolerance=pd.Timedelta('120D'))
        merged_list.append(merged)
    merged = pd.concat(merged_list).reset_index()
    print(f"   Merged: {len(merged)} rows")
    return merged

def engineer_features_3m(df):
    print("Engineering features + 3-month target...")
    g = df.groupby('Ticker')
    
    df['ret_1m']  = g['price_close'].pct_change(1)
    df['ret_3m']  = g['price_close'].pct_change(3)
    df['ret_6m']  = g['price_close'].pct_change(6)
    df['ret_12m'] = g['price_close'].pct_change(12)
    df['vol_6m']  = g['ret_1m'].rolling(6, min_periods=4).std().reset_index(0, drop=True)

    df['pe'] = df['price_close'] * 1e6 / df['netIncome'].abs().replace(0, np.nan)
    df['pb'] = df['price_close'] * 1e6 / df['totalShareholderEquity'].replace(0, np.nan)
    df['de_ratio'] = df['totalDebt'] / df['totalShareholderEquity'].replace(0, np.nan)
    df['roa'] = df['netIncome'] / df['totalAssets'].replace(0, np.nan)
    df['roe'] = df['netIncome'] / df['totalShareholderEquity'].replace(0, np.nan)

    for col in ['netIncome','totalRevenue','totalAssets','totalDebt','totalShareholderEquity','operatingCashflow','pe','pb']:
        df[f'{col}_lag1'] = g[col].shift(1)
        df[f'{col}_growth'] = g[col].pct_change(1)

    df['future_price'] = g['price_close'].shift(-FORWARD_MONTHS)
    df['future_return_3m'] = df['future_price'] / df['price_close'] - 1
    df[TARGET_NAME] = (df['future_return_3m'] > 0).astype(int)

    df = df.dropna(subset=[TARGET_NAME])
    print(f"   Features engineered: {len(df)} rows")
    return df

def add_macro_features(df):
    print("Adding macro features (VIX, 10Y yield)...")
    if not os.path.exists(MACRO_PATH):
        print("   Macro file not found — skipping macro")
        return df
    
    macro = pd.read_csv(MACRO_PATH, sep=';', decimal=',', parse_dates=['Date'])
    df = df.merge(macro, on='Date', how='left')
    
    macro_cols = [col for col in macro.columns if col != 'Date']
    df[macro_cols] = df.groupby('Ticker')[macro_cols].ffill()
    
    print(f"   Macro columns added: {macro_cols}")
    return df

def feature_ranking_and_selection(df, final_features):
    print("Feature ranking with LightGBM...")
    
    # Use early period for ranking to avoid leakage
    train_rank = df[df['Date'] < '2019-01-01'].copy()
    if len(train_rank) == 0:
        print("   No early data for ranking — skipping")
        return df, final_features
    
    X_rank = train_rank[final_features]
    y_rank = train_rank[TARGET_NAME]
    
    lgb_rank = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    lgb_rank.fit(X_rank, y_rank)
    
    importances = pd.Series(lgb_rank.feature_importances_, index=final_features).sort_values(ascending=False)
    
    # Select top 50 features
    top_n = 50
    top_features = importances.head(top_n).index.tolist()
    print(f"   Selected top {len(top_features)} features (from {len(final_features)})")
    
    # Apply to full df
    keep_cols = ['Date', 'Ticker', TARGET_NAME] + top_features
    df = df[keep_cols].copy()
    
    return df, top_features

def clean_winsorize_scale(df):
    feature_cols = [c for c in df.columns if c not in ['Date','Ticker',TARGET_NAME,'future_price','future_return_3m']]
    df[feature_cols] = df.groupby('Date')[feature_cols].transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
    df[feature_cols] = df.groupby('Ticker')[feature_cols].ffill().fillna(df[feature_cols].median())
    
    scaler = StandardScaler()
    df[feature_cols] = df.groupby('Date')[feature_cols].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten())
    return df, feature_cols

def split_and_save(df, features):
    print("Dynamic time-series split (70% train, 15% val, 15% test)...")
    
    # Sort by Date to ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]
    
    print(f"   Train: {len(train)} rows (up to {train['Date'].max().date()})")
    print(f"   Val:   {len(val)} rows ({val['Date'].min().date()} to {val['Date'].max().date()})")
    print(f"   Test:  {len(test)} rows (from {test['Date'].min().date()} onward)")
    
    train.to_csv(f'{PROCESSED_DIR}/train_3m.csv', index=False)
    val.to_csv(f'{PROCESSED_DIR}/val_3m.csv', index=False)
    test.to_csv(f'{PROCESSED_DIR}/test_3m.csv', index=False)
    with open(f'{PROCESSED_DIR}/features_3m.txt', 'w') as f:
        f.write('\n'.join(features))

    print(f"\n3-month datasets saved dynamically")

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    df = merge_asof(load_prices(), load_fundamentals())
    df = engineer_features_3m(df)
    df = add_macro_features(df)  # <-- Macro added
    df, feats = clean_winsorize_scale(df)
    df, final_features = feature_ranking_and_selection(df, feats)  # <-- Ranking added
    
    # Note: Threshold optimization in prediction scripts
    # Note: Portfolio ranking in evaluation (long top decile, short bottom)
    
    split_and_save(df, final_features)

if __name__ == '__main__':
    main()
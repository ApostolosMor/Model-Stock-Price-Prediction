# updating_data_and_models.py
# Single script to update data up to a given date, preprocess, retrain all models + stacking

import os
import sys
import pandas as pd
import yfinance as yf
import requests
import time
import joblib
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ========================= CONFIG =========================
TICKER_FILE = 'tickers.txt'
RAW_PRICES_PATH = 'data/raw/raw_stock_prices.csv'
RAW_FUNDS_PATH = 'data/raw/raw_fundamentals_av.csv'
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
STACKED_DIR = 'models/stacked'
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STACKED_DIR, exist_ok=True)

AV_KEY = 'ZO33G3099I1HZVV9'
START_DATE = '2005-01-01'
INTERVAL = '1mo'

HORIZONS = ['3m', '6m']

# Base models for stacking (LSTM excluded for reliability)
BASE_MODEL_NAMES = ['Logistic', 'Random Forest', 'XGBoost', 'LightGBM']

print("=== DATA & MODEL UPDATE SCRIPT ===\n")

# 1. Ask for end date
end_date_input = input("Enter end date for price data (format DD/MM/YYYY, e.g. 20/11/2025): ").strip()
try:
    END_DATE = pd.to_datetime(end_date_input, format='%d/%m/%Y').strftime('%Y-%m-%d')
    print(f"Using end date: {END_DATE}\n")
except:
    print("Invalid date format — exiting.")
    sys.exit(1)

# 2. Fetch prices
print("Fetching monthly prices...")
with open(TICKER_FILE, 'r') as f:
    ticker_list = [line.strip() for line in f if line.strip()]

data = yf.download(
    ticker_list,
    start=START_DATE,
    end=END_DATE,
    interval=INTERVAL,
    progress=False
)

prices_df = data['Close'].stack().reset_index()
prices_df.columns = ['Date', 'Ticker', 'Close']
prices_df['Close'] = prices_df['Close'].round(6)
prices_df.to_csv(RAW_PRICES_PATH, index=False, sep=';', decimal=',')

print(f"Prices updated and saved ({len(prices_df)} rows)\n")

# 3. Fetch fundamentals (unchanged — quarterly up to latest)
print("Fetching fundamentals (this may take 20–30 minutes due to API limits)...")
# (same code as your fetch_fundamentals — pasted inline)

tickers = ticker_list

AV_FUNCTIONS = {
    'INCOME': 'INCOME_STATEMENT',
    'BALANCE': 'BALANCE_SHEET',
    'CASHFLOW': 'CASH_FLOW'
}

def fetch_quarterly_data(ticker, function_name):
    url = 'https://www.alphavantage.co/query'
    params = {'function': function_name, 'symbol': ticker, 'apikey': AV_KEY}
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        if 'quarterlyReports' not in data:
            return None
        df = pd.DataFrame(data['quarterlyReports'])
        df['Date'] = pd.to_datetime(df['fiscalDateEnding'], dayfirst=True)
        df = df.drop(columns=['fiscalDateEnding'], errors='ignore')
        return df
    except:
        return None

all_data = []
for i, t in enumerate(tickers, 1):
    print(f"[{i}/{len(tickers)}] {t}...", end="")
    dfs = []
    for func in AV_FUNCTIONS.values():
        df = fetch_quarterly_data(t, func)
        if df is not None and not df.empty:
            dfs.append(df)
        time.sleep(12)
    if len(dfs) == 3:
        merged = pd.concat(dfs, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merged['Ticker'] = t
        all_data.append(merged)
        print(" OK")
    else:
        print(" skipped")

if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    cols = ['Date', 'Ticker', 'netIncome', 'totalRevenue', 'totalAssets',
            'longTermDebt', 'shortLongTermDebtTotal', 'totalShareholderEquity',
            'totalCurrentAssets', 'totalCurrentLiabilities', 'operatingCashflow']
    available = [c for c in cols if c in full_df.columns]
    full_df = full_df[available].copy()
    full_df['totalDebt'] = full_df.get('longTermDebt', 0).fillna(0) + full_df.get('shortLongTermDebtTotal', 0).fillna(0)
    full_df['currentAssets'] = full_df.get('totalCurrentAssets', 0)
    full_df['currentLiabilities'] = full_df.get('totalCurrentLiabilities', 0)
    final_cols = ['Date', 'Ticker', 'netIncome', 'totalRevenue', 'totalAssets',
                  'totalDebt', 'totalShareholderEquity', 'currentAssets',
                  'currentLiabilities', 'operatingCashflow']
    full_df = full_df[final_cols]
    full_df.to_csv(RAW_FUNDS_PATH, index=False, sep=';', decimal=',')
    print(f"\nFundamentals saved ({len(full_df)} rows)\n")
else:
    print("\nNo fundamentals fetched — using existing file.\n")

# 4. Run preprocessing (assume your preprocess script is preprocess_data_3month.py and 6month.py — run both)
print("Running preprocessing for 3m and 6m horizons...")
os.system('python scripts/preprocess_data_3month.py')
os.system('python scripts/preprocess_data_6month.py')
print("Preprocessing complete.\n")

# 5. Retrain individual models (Logistic, RF, XGBoost, LightGBM)
print("Retraining individual models...\n")
# (use your existing training scripts)
os.system('python scripts/train_logistic_models.py')
os.system('python scripts/train_random_forest.py')
os.system('python scripts/train_xgboost.py')
os.system('python scripts/train_lightgbm.py')
print("Individual models retrained.\n")

# 6. Retrain stacking
print("Retraining stacking ensemble...\n")
os.system('python scripts/train_stacking.py')
print("Stacking retrained.\n")

print("=== UPDATE COMPLETE ===")
print(f"Data updated to {END_DATE}")
print("All models and stacking ensemble retrained.")
print("You can now use predict_ticker_interactive.py (blending) or predict_stacking.py (stacking) for real-time predictions.")
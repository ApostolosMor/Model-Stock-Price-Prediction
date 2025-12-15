# ← Paste this entire script and run
import requests
import pandas as pd
import os
import time

AV_KEY = 'ZO33G3099I1HZVV9'
TICKER_FILE = 'tickers.txt'
OUTPUT_FILE_PATH = 'data/raw/raw_fundamentals_av.csv'

AV_FUNCTIONS = {
    'INCOME': 'INCOME_STATEMENT',
    'BALANCE': 'BALANCE_SHEET',
    'CASHFLOW': 'CASH_FLOW'
}

def load_tickers():
    with open(TICKER_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def fetch_quarterly_data(ticker, function_name):
    url = 'https://www.alphavantage.co/query'
    params = {'function': function_name, 'symbol': ticker, 'apikey': AV_KEY}
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        if 'quarterlyReports' not in data:
            return None
        df = pd.DataFrame(data['quarterlyReports'])
        df['Date'] = pd.to_datetime(df['fiscalDateEnding'])
        df = df.drop(columns=['fiscalDateEnding'], errors='ignore')
        return df
    except:
        return None

def fetch_fundamentals():
    tickers = load_tickers()
    print(f"Fetching 20+ years quarterly data for {len(tickers)} tickers...")

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

    if not all_data:
        print("No data.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # Final cleanup
    cols = ['Date', 'Ticker', 'netIncome', 'totalRevenue', 'totalAssets',
            'longTermDebt', 'shortLongTermDebtTotal', 'totalShareholderEquity',
            'totalCurrentAssets', 'totalCurrentLiabilities', 'operatingCashflow']
    available = [c for c in cols if c in full_df.columns]
    full_df = full_df[available].copy()

    # Normalize names
    full_df['totalDebt'] = full_df.get('longTermDebt', 0).fillna(0) + full_df.get('shortLongTermDebtTotal', 0).fillna(0)
    full_df['currentAssets'] = full_df.get('totalCurrentAssets', 0)
    full_df['currentLiabilities'] = full_df.get('totalCurrentLiabilities', 0)

    final_cols = ['Date', 'Ticker', 'netIncome', 'totalRevenue', 'totalAssets',
                  'totalDebt', 'totalShareholderEquity', 'currentAssets',
                  'currentLiabilities', 'operatingCashflow']
    full_df = full_df[final_cols]

    os.makedirs('data/raw', exist_ok=True)
    full_df.to_csv(OUTPUT_FILE_PATH, index=False, sep=';', decimal=',')

    print(f"\nSUCCESS: {len(full_df)} records saved!")
    print(full_df.head())

if __name__ == '__main__':
    fetch_fundamentals()
import pandas as pd
import os

# INPUT FILES
PRICES_FILE = 'data/raw/raw_stock_prices.csv'
FUNDAMENTALS_FILE = 'data/raw/raw_fundamentals_av.csv'
OUTPUT_FILE = 'data/processed/FINAL_COMPLETE_DATASET.csv'

print("Loading data...")
prices = pd.read_csv(PRICES_FILE, sep=';', decimal=',')
fund = pd.read_csv(FUNDAMENTALS_FILE, sep=';', decimal=',')

prices['Date'] = pd.to_datetime(prices['Date'], dayfirst=True)
fund['Date'] = pd.to_datetime(fund['Date'], dayfirst=True)

# CRITICAL: Drop rows with NaT Date in fundamentals (PYPL fix)
fund = fund.dropna(subset=['Date'])

# Convert all to numeric
cols = ['netIncome', 'totalRevenue', 'totalAssets', 'totalDebt',
        'totalShareholderEquity', 'currentAssets', 'currentLiabilities',
        'operatingCashflow', 'Close']
for col in cols:
    if col in prices.columns:
        prices[col] = pd.to_numeric(prices[col], errors='coerce')
    if col in fund.columns:
        fund[col] = pd.to_numeric(fund[col], errors='coerce')

print(f"Prices: {len(prices)} | Fundamentals: {len(fund)} rows")

all_merged = []
for ticker in prices['Ticker'].unique():
    print(f"Processing {ticker}...", end=" ")
    
    p = prices[prices['Ticker'] == ticker][['Date', 'Ticker', 'Close']].sort_values('Date')
    f = fund[fund['Ticker'] == ticker].sort_values('Date')
    
    if f.empty:
        print("no data")
        continue
    
    m = pd.merge_asof(
        p,
        f,
        on='Date',
        by='Ticker',
        direction='backward',
        tolerance=pd.Timedelta('120 days')
    )
    
    cols = ['netIncome', 'totalRevenue', 'totalAssets', 'totalDebt',
            'totalShareholderEquity', 'currentAssets', 'currentLiabilities',
            'operatingCashflow']
    m[cols] = m[cols].ffill()
    
    all_merged.append(m)
    print("OK")

final_df = pd.concat(all_merged, ignore_index=True)

# RATIOS
final_df['ROE'] = final_df['netIncome'] / final_df['totalShareholderEquity']
final_df['ROA'] = final_df['netIncome'] / final_df['totalAssets']
final_df['Debt_Ratio'] = final_df['totalDebt'] / final_df['totalAssets']
final_df['Debt_to_Equity'] = final_df['totalDebt'] / final_df['totalShareholderEquity']
final_df['Current_Ratio'] = final_df['currentAssets'] / final_df['currentLiabilities']
final_df['Net_Profit_Margin'] = final_df['netIncome'] / final_df['totalRevenue']
final_df['Operating_Margin'] = final_df['operatingCashflow'] / final_df['totalRevenue']

# Market ratios
final_df['EPS_Annual'] = final_df['netIncome'] * 4
final_df['Market_PE'] = final_df['Close'] / final_df['EPS_Annual'].replace(0, pd.NA)
final_df['Market_PB'] = final_df['Close'] / (final_df['totalShareholderEquity'] / 1e9)
final_df['Market_PS'] = (final_df['Close'] * 1e9) / final_df['totalRevenue'].replace(0, pd.NA)

cols = ['Date', 'Ticker', 'Close', 'Market_PE', 'Market_PB', 'Market_PS',
        'ROE', 'ROA', 'Debt_Ratio', 'Debt_to_Equity', 'Current_Ratio',
        'Net_Profit_Margin', 'Operating_Margin']
final_df = final_df[cols]

os.makedirs('data/processed', exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False, sep=';', decimal=',')

print(f"\nSUCCESS! {len(final_df):,} rows saved to {OUTPUT_FILE}")
print("Sample:")
print(final_df.head(10))
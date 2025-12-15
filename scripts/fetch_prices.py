import yfinance as yf
import pandas as pd
import os

# --- Configuration ---
# Define the file paths based on your project structure
TICKER_FILE = 'tickers.txt'
OUTPUT_FILE_PATH = 'data/raw/raw_stock_prices.csv'
START_DATE = '2005-01-01'
END_DATE = '2025-01-01'
INTERVAL = '1mo' # Monthly data

def fetch_data():
    """Reads tickers, downloads monthly price data, and saves it to a CSV."""
    
    # 1. Load Tickers
    try:
        with open(TICKER_FILE, 'r') as f:
            # Strip whitespace and ignore empty lines
            ticker_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(ticker_list)} tickers from {TICKER_FILE}")
    except FileNotFoundError:
        print(f" Error: {TICKER_FILE} not found. Ensure it's in the project root directory.")
        return
    
    # 2. Download Data using yfinance
    print(f"⏳ Downloading monthly data ({START_DATE} to {END_DATE})... This may take a few minutes.")
    
    try:
        # yfinance.download can take a list of tickers
        data = yf.download(
            ticker_list,
            start=START_DATE,
            end=END_DATE,
            interval=INTERVAL,
            progress=False # Set to True to see individual download progress
        )
        print(" Download complete.")
    except Exception as e:
        print(f" An error occurred during download: {e}")
        return

    # 3. Clean and Save Data
    # Extract only the 'Close' price and restructure (stack) the DataFrame
    prices_df = data['Close'].stack().reset_index()
    
    # Rename the columns for clarity and easier merging later
    prices_df.columns = ['Date', 'Ticker', 'Close']
    
    # Round the Close price to 6 decimal places (more than enough)
    prices_df['Close'] = prices_df['Close'].round(6)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    
    # Save the raw data
    # prices_df.to_csv(OUTPUT_FILE_PATH, index=False)
    prices_df.to_csv(OUTPUT_FILE_PATH, index=False, sep=';', decimal=',')
    
    print(f"\n Successfully saved {len(prices_df):,} records to {OUTPUT_FILE_PATH}")
    print(f"First 5 rows of the raw data:\n{prices_df.head()}")

if __name__ == '__main__':
    fetch_data()
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load models
logreg = joblib.load('models/logistic_model.pkl')
rf = joblib.load('models/random_forest_model.pkl')
xgb = joblib.load('models/xgboost_model.pkl')
lstm = load_model('models/lstm_model.h5')

# Load data
df = pd.read_csv('data/processed/PREPROCESSED_FOR_MODELING.csv', sep=';', decimal=',')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

features = ['Market_PE', 'Market_PB', 'Market_PS', 'ROE', 'ROA',
            'Debt_Ratio', 'Debt_to_Equity', 'Current_Ratio',
            'Net_Profit_Margin', 'Operating_Margin']

def predict_all(ticker):
    ticker = ticker.upper().strip()
    if ticker not in df['Ticker'].unique():
        print(f"ERROR: {ticker} not in dataset")
        return
    
    # Get latest row
    row = df[df['Ticker'] == ticker].sort_values('Date').iloc[-1]
    X_single = row[features].values.reshape(1, -1)
    
    # Predictions
    lr = "RISE" if logreg.predict(X_single)[0] == 1 else "FALL"
    lr_prob = logreg.predict_proba(X_single)[0][1]
    
    rf_pred = "RISE" if rf.predict(X_single)[0] == 1 else "FALL"
    rf_prob = rf.predict_proba(X_single)[0][1]
    
    xgb_pred = "RISE" if xgb.predict(X_single)[0] == 1 else "FALL"
    xgb_prob = xgb.predict_proba(X_single)[0][1]
    
    # LSTM sequence
    seq_df = df[df['Ticker'] == ticker].tail(24)[features]
    seq = seq_df.values.reshape(1, 24, -1)
    lstm_prob = lstm.predict(seq, verbose=0)[0][0]
    lstm_pred = "RISE" if lstm_prob > 0.5 else "FALL"
    
    # Print beautiful result
    print(f"\n{ticker} — 3-Month Prediction")
    print(f"Date: {row['Date'].date()} | Price: ${row['Close']:.2f}")
    print("="*60)
    print(f"{'Model':<20} {'Prediction':<8} {'Confidence'}")
    print("-"*60)
    print(f"{'Logistic Reg':<20} {lr:<8} {lr_prob:6.1%}")
    print(f"{'Random Forest':<20} {rf_pred:<8} {rf_prob:6.1%}")
    print(f"{'XGBoost':<20} {xgb_pred:<8} {xgb_prob:6.1%}")
    print(f"{'LSTM':<20} {lstm_pred:<8} {lstm_prob:6.1%}")
    
    # Majority vote
    votes = [lr, rf_pred, xgb_pred, lstm_pred].count("RISE")
    final = "RISE ↑↑↑" if votes >= 3 else "RISE ↑↑" if votes == 2 else "FALL ↓↓" if votes <=1 else "FALL ↓"
    print(f"\n→ MAJORITY VOTE ({votes}/4): {final}")

# Run
if __name__ == "__main__":
    t = input("\nEnter ticker: ").strip()
    predict_all(t)
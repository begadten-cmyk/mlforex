import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd  # Added for column manipulation

# Fetch GBP/JPY forex data (1-minute intervals, last 1 day)
gbp_jpy_data = yf.download(tickers='GBPJPY=X', interval='1m', period='1d')

# Reset index and flatten column names
gbp_jpy_data.reset_index(inplace=True)
gbp_jpy_data.columns = [str(col) for col in gbp_jpy_data.columns]  # Convert all column names to strings

# Drop any rows with NaN values initially to avoid calculation issues
gbp_jpy_data.dropna(inplace=True)

# Print to verify structure
print("Initial Data:")
print(gbp_jpy_data.head())

# Calculate the Average True Range (ATR)
ATR_PERIOD = 14

# Add True Range columns
gbp_jpy_data['High-Low'] = gbp_jpy_data['High'] - gbp_jpy_data['Low']
gbp_jpy_data['High-Close'] = abs(gbp_jpy_data['High'] - gbp_jpy_data['Close'].shift(1))
gbp_jpy_data['Low-Close'] = abs(gbp_jpy_data['Low'] - gbp_jpy_data['Close'].shift(1))

# Calculate the True Range (TR)
gbp_jpy_data['True Range'] = gbp_jpy_data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)

# Calculate the ATR
gbp_jpy_data['ATR'] = gbp_jpy_data['True Range'].rolling(window=ATR_PERIOD).mean()

# Drop NaN values caused by rolling calculations
gbp_jpy_data.dropna(subset=['ATR'], inplace=True)

print("\nData with ATR calculated:")
print(gbp_jpy_data[['High', 'Low', 'Close', 'True Range', 'ATR']].head(10))

from datetime import datetime

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


# ticker parameters
ticker = ("AAPL")
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)

short_sma = 20
long_sma = 50

# fetching data
data = yf.download(ticker, start=start_date, end=end_date,)
close_aapl = data[('Close', 'AAPL')]
close_aapl.plot(title="AAPL Close Price")
data.drop(columns=["Volume", "High", "Low", "Open"], inplace=True)

print(data.head())

#gen sma
data["SMA20"] = data[('Close')].rolling(window=short_sma).mean()
data["SMA50"] = data[('Close')].rolling(window=long_sma).mean()

print(data)
data.plot(title="test")
plt.show()

#gen signal
data['Signal'] = 0
data.loc[data['SMA20'] > data['SMA50'], 'Signal'] = 1
data.loc[data['SMA20'] < data['SMA50'], 'Signal'] = 0
data['Position'] = data['Signal'].shift(1)

 #bt logic
data['Return'] = data[('Close')].pct_change()
data['Strategy'] = data['Position'] * data['Return']
data.dropna(inplace=True)

 #retruns calc
cumulative_returns = (1 + data[['Return', 'Strategy']]).cumprod()

print(cumulative_returns)



plt.figure(figsize=(12,6))
plt.plot(cumulative_returns['Return'], label='Buy & Hold', color='black')
plt.plot(cumulative_returns['Strategy'], label='SMA Strategy', color='green')
plt.title(f"SMA Crossover Strategy on {ticker}")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





import talib
import pandas as pd


def process_data(eod_data):

    # Create a pandas data frame
    data = pd.DataFrame()

    # Copy original data
    data['H'] = eod_data['High']
    data['L'] = eod_data['Low']
    data['O'] = eod_data['Open']
    data['C'] = eod_data['Close']
    data['V'] = eod_data['Volume']

    # Calculate indicators
    data['H-L'] = eod_data['High'] - eod_data['Low']
    data['C-O'] = eod_data['Close'] - eod_data['Open']
    data['3day SMA'] = eod_data['Close'].shift(1).rolling(window=3).mean()
    data['10day SMA'] = eod_data['Close'].shift(1).rolling(window=10).mean()
    data['30day SMA'] = eod_data['Close'].shift(1).rolling(window=30).mean()
    data['Std_dev'] = eod_data['Close'].rolling(5).std()
    data['RSI'] = talib.RSI(eod_data['Close'].values, 9)

    # Generate boolean containing flag if the price rose
    data['Rolling Skewness'] = pd.DataFrame(eod_data['Close']).rolling(30).skew().values

    return data.dropna()

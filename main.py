import numpy as np
import random

from load_stock_data import load_stock_data
from process_data import process_data
from run_neural_network import run_neural_network
from evaluate import evaluate
from remap import remap
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

TEST_SET = 0.9
WINDOW = 30
STEP = 1
FORECAST = 1
ROLLING = 30
EMB_SIZE = 7

random.seed(42)

# Load stock data
eod_data = load_stock_data()

# Loop through all stocks returned
for key, data in eod_data.items():

    # Process data
    processed_data = process_data(data)

    X, Y = [], []
    for idx in range(0, len(processed_data)-WINDOW-FORECAST, STEP):

        # Get data from window
        hl = remap(np.array(processed_data['H-L'][idx:idx+WINDOW]), -1, 1)
        co = remap(np.array(processed_data['C-O'][idx:idx+WINDOW]), -1, 1)
        sma_3 = remap(np.array(processed_data['3day SMA'][idx:idx+WINDOW]), -1, 1)
        sma_10 = remap(np.array(processed_data['10day SMA'][idx:idx+WINDOW]), -1, 1)
        sma_30 = remap(np.array(processed_data['30day SMA'][idx:idx+WINDOW]), -1, 1)
        std_dev = remap(np.array(processed_data['Std_dev'][idx:idx+WINDOW]), -1, 1)
        rsi = remap(np.array(processed_data['RSI'][idx:idx+WINDOW]), -1, 1)

        # Stack in array
        x_i = np.column_stack((hl, co, sma_3, sma_10, sma_30, std_dev, rsi))

        # Get forecast value
        last_close = processed_data['C'][idx + WINDOW]
        next_close = processed_data['C'][idx + WINDOW + FORECAST]

        if last_close < next_close:
            y_i = [1, 0]
        else:
            y_i = [0, 1]

        # Add to data set
        X.append(x_i)
        Y.append(y_i)

    # Create numpy arrays
    X, Y = np.array(X), np.array(Y)

    # Split the data set into train and test data sets
    split = int(len(processed_data) * TEST_SET)
    X_train, X_test, Y_train, Y_test = X[:split], X[split:], Y[:split], Y[split:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))

    # Run neural network
    history, prediction = run_neural_network(X_train, Y_train, X_test, Y_test)
    original = Y_test

    evaluate(history, prediction, original)

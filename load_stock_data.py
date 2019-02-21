import os
import pickle
import pandas_datareader as pdr


def load_stock_data():

    # List of stocks to collect data from
    stocks = ['INVE-B.ST']

    # Name of file to save data within
    file_name = "eod_data.p"

    # Load file if it exists, otherwise, collect data from yahoo
    if os.path.exists(file_name):

        # Load from pickle file
        eod_data = pickle.load(open(file_name, "rb"))
    else:

        # Download end of day data
        eod_data = {}
        for stock in stocks:
            eod_data[stock] = pdr.get_data_yahoo(stock)

        # Save to pickle file
        pickle.dump(eod_data, open(file_name, "wb"))

    # Return list of stock data
    return eod_data

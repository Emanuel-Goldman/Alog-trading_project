import matplotlib
import pandas as pd
import os
import requests
import numpy as np
from datetime import datetime
from enum import Enum
from typing import List
import matplotlib.pyplot as plt
from strategies import BaseStrategy as bs
from models import *
from backtesting import *
from evaluation import *

def make_api_call(base_url, endpoint="", method="GET", **kwargs):
    # Construct the full URL
    full_url = f'{base_url}{endpoint}'

    # Make the API call
    response = requests.request(method=method, url=full_url, **kwargs)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        return response
    else:
        # If the request was not successful, raise an exception with the error message
        raise Exception(f'API request failed with status code {response.status_code}: {response.text}')
    

def get_binance_historical_data_old(symbol, interval, start_date, end_date=None):
    
    # define basic parameters for call
    base_url = 'https://fapi.binance.com'
    endpoint = '/fapi/v1/klines'
    method = 'GET'
    
    # Set the start time parameter in the params dictionary
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1500,
        'startTime': start_date, # Start time in milliseconds
        'endTime': end_date if end_date else 9999999999999
    }


    # Make initial API call to get candles
    response = make_api_call(base_url, endpoint=endpoint, method=method, params=params)

    candles_data = []

    while len(response.json()) > 0 :
        # Append the received candles to the list
        candles_data.extend(response.json())

        # Update the start time for the next API call
        params['startTime'] = candles_data[-1][0] + 1 # last candle open_time + 1ms
        # Make the next API call
        response = make_api_call(base_url, endpoint=endpoint, method=method, params=params)

    
    # Wrap the candles data as a pandas DataFrame
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    dtype={
    'open_time': 'datetime64[ms, Asia/Jerusalem]',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
    'close_time': 'datetime64[ms, Asia/Jerusalem]',
    'quote_asset_volume': 'float64',
    'number_of_trades': 'int64',
    'taker_buy_base_asset_volume': 'float64',
    'taker_buy_quote_asset_volume': 'float64',
    'ignore': 'float64'
    }
    
    df = pd.DataFrame(candles_data, columns=columns)
    df = df.astype(dtype)

    return df


def get_binance_historical_data(symbol, interval, start_date, end_date=None):
    
    # define basic parameters for call
    base_url = 'https://fapi.binance.com'
    endpoint = '/fapi/v1/klines'
    method = 'GET'
    
    # Set the start time parameter in the params dictionary
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1500,
        'startTime': start_date, # Start time in milliseconds
        'endTime': end_date 
    }


    # Make initial API call to get candles
    response = make_api_call(base_url, endpoint=endpoint, method=method, params=params)

    candles_data = []

    while len(response.json()) > 0 :
        # Append the received candles to the list
        candles_data.extend(response.json())

        # Update the start time for the next API call
        params['startTime'] = candles_data[-1][0] + 1 # last candle open_time + 1ms
        # Make the next API call
        response = make_api_call(base_url, endpoint=endpoint, method=method, params=params)

    
    # Wrap the candles data as a pandas DataFrame
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    dtype={
    'open_time': 'datetime64[ms, Asia/Jerusalem]',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
    'close_time': 'datetime64[ms, Asia/Jerusalem]',
    }
    
    
    df = pd.DataFrame(candles_data, columns=columns)
    df.drop(['quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1, inplace=True)
    df = df.astype(dtype)

    return df


def is_in_range(date):
    return int(date.timestamp() * 1000) 


def plot_strategy(df):
    plt.figure(figsize= (25,10))
    plt.plot(df['close'], label = 'Price')
    # Trade Signs:
    plt.scatter(df.index, df['buy'], label = 'Buy',alpha = 1, marker = '^', color = 'green')
    plt.scatter(df.index, df['sell'], label = 'Sell', alpha = 1, marker = 'v', color = 'red')
    
    plt.legend()
    plt.ylabel('Price')
    plt.title('Price with Buy and Sell Signals')   
    plt.show()


def plot_profit(df):
    plt.figure(figsize=(25,10))
    plt.plot(df['portfolio_value'].pct_change(1)*100, label = 'Portfolio')
    plt.legend()
    plt.show()


def get_csv_data(symbol, interval, start_date, end_date,name):
    btcusdt_df = get_binance_historical_data(symbol, interval, start_date)
    in_range_btcusdt_df = btcusdt_df[btcusdt_df['open_time'].apply(is_in_range) < end_date]
    in_range_btcusdt_df.to_csv('data_for_part_2.csv', index=False)  
    return in_range_btcusdt_df


def get_all_data(coins, interval, start_date, end_date):
    for coin in coins:
        df = get_binance_historical_data(coin, interval, start_date, end_date)
        df.to_csv(f'data/{coin}_{interval}.csv', index=False)
        print(f'{coin} is done')
    print('All the data is ready')
    
    
def calculate_returns_column(df):
    df['returns'] = df['close'].pct_change(1)



def main():
    
    coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT',
             'ADAUSDT', 'XRPUSDT', 'DOGEUSDT',
             'DOTUSDT', 'UNIUSDT', 'LINKUSDT',
             'LTCUSDT', 'BCHUSDT', 'SOLUSDT',
             'MATICUSDT', 'XLMUSDT', 'ETCUSDT',
             'THETAUSDT', 'VETUSDT', 'TRXUSDT',
             'EOSUSDT', 'FILUSDT', 'AAVEUSDT',
             'XTZUSDT', 'ATOMUSDT', 'NEOUSDT',
             'ALGOUSDT', 'MKRUSDT', 'COMPUSDT',
             'KSMUSDT']
    # we can also use the coins 'CROUSDT', 'HTUSDT','CHZUSDT', 'SNXUSDT', 'YFI' if we want to add more coins
    
    files =  os.listdir('data')
    
    for file in files:
        df = pd.read_csv(f'data/{file}')
        calculate_returns_column(df)
        df.to_csv(f'data/{file}', index=False)
    
    # Todo list:
    # 1. To get all the data from the binance API to a csv file
    # 2. To clean the data
    # 3. To create the model that predicts 
    # 4. To create the strategy
    # 5. To backtest the strategy
    

if __name__ == "__main__":
    main()
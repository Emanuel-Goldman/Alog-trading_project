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
    
    
def is_in_range(date):
    return int(date.timestamp() * 1000)


def calculate_returns_column(df):
    df['returns'] = df['close'].pct_change(1) 
    
    
def clean_data():
    for file in files:
        df_curr = pd.read_csv(f'data/{file}')
        df_curr = df_curr.astype(dtype)
        df_curr = df_curr[(df_curr['open_time'] < '2024-06-16 16:00:00+03:00') & (df_curr['open_time'] > '2020-10-23 10:00:00+03:00')]
        print(f'{file}, its shape is {df_curr.shape[0]}, first date is {df_curr.iloc[0]["open_time"]}, last date is {df_curr.iloc[-1]["open_time"]}')
        df_curr.to_csv(f'data/{file}', index=False)
        
def combine_data(column, coins):    
    files =  os.listdir('data')
    coins.sort()
    df = pd.DataFrame(columns=coins)
    for file, coin in zip(files, coins):
        print(f'file: {file} coin: {coin}')
        df_curr = pd.read_csv(f'data/{file}')
        df[f'{coin}'] = df_curr[f'{column}']
    
    df.to_csv(f'result/{column}.csv', index=False)
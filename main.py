import matplotlib
import pandas as pd
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
    

def get_binance_historical_data(symbol, interval, start_date):
    
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


def is_in_range(date):
    return int(date.timestamp() * 1000) 


def ATR(data , atr_length):
    data['tr'] = abs(np.maximum(data['high'], data['close'].shift(1)) - np.minimum(data['low'], data['close'].shift(1)))
    data['atr'] = data['tr'].rolling(window=atr_length).mean()
    return data


def UTBot(data, key_value: int, atr_length: int) -> List[int]:

    ATR(data, atr_length)

    data['loss_threshold'] = data['atr'] * key_value

    data['trailing_stop'] = 0

    close = data['close'].tolist()
    trailStop = [0] * len(close)
    trailStop[0] = close[0]
    lossThrsh = data['loss_threshold'].tolist()

    for i in range(1, len(trailStop)):
        if (close[i] > trailStop[i - 1]) & (close[i - 1] > trailStop[i - 1]):
            trailStop[i] = max(trailStop[i - 1], close[i] - lossThrsh[i])
        elif (close[i] < trailStop[i - 1]) & (close[i - 1] < trailStop[i - 1]):
            trailStop[i] = min(trailStop[i - 1], close[i] + lossThrsh[i])
        elif (close[i] > trailStop[i - 1]):
            trailStop[i] = close[i] - lossThrsh[i]
        else:
            trailStop[i] = close[i] + lossThrsh[i]

    data['trailing_stop'] = trailStop
    return trailStop


def cross(a,b,prev_a,prev_b):
    return a > b and prev_a < prev_b


def buy_signal(trailing_stop:List[int],close:List[int])->List[bool]:

    buy_signal = [False]*len(trailing_stop)
    # We would buy the coin if we crossed the trailing stop from below - the close turned 
    # bigger then the trailing stop
    for i in range(1,len(trailing_stop)):
        buy_signal[i] = cross(a=close[i],b=trailing_stop[i],prev_a=close[i-1],prev_b=trailing_stop[i-1])

    return buy_signal


def sell_signal(trailing_stop:List[int],close:List[int])->List[bool]:

    sell_signal = [False]*len(trailing_stop)
    # We would sell the coin if we crossed the trailing stop from above - the close turned 
    # smaller then the trailing stop
    for i in range(1,len(trailing_stop)):
        sell_signal[i] = cross(a=trailing_stop[i],b=close[i],prev_a=trailing_stop[i-1],prev_b=close[i-1])
    
    return sell_signal


def EMA(df,fast_len,slow_len):
    df['EMA_fast'] = df['close'].ewm(span=fast_len, min_periods=fast_len, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow_len, min_periods=slow_len, adjust=False).mean()
    return df  


def MacD(df, fast_length:int, slow_length:int):
    df = EMA(df,fast_length,slow_length)
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    return df['MACD']


def SmoothSrs(srs, smoothFact):
    smoothed_srs = srs.copy()
    for i in range(1, len(smoothed_srs)):
        if np.isnan(smoothed_srs[i-1]):
            smoothed_srs[i] = srs[i]
        else:
            smoothed_srs[i] = smoothed_srs[i-1] + smoothFact * (srs[i] - smoothed_srs[i-1])
    return smoothed_srs


def NormalizeSmoothSrs(df ,srs, window, smoothFact):
    # finding the lowest and highest range
    lowest = srs.rolling(window).min()
    highestRange = srs.rolling(window).max() - lowest
    
    # normalizing srs
    normalizedsrs = srs.copy()
    normalizedsrs[highestRange > 0] = ((srs - lowest) / highestRange * 100)*(highestRange > 0)
    normalizedsrs[highestRange <= 0] = np.nan
    normalizedsrs.ffill(inplace = True)
    
    # smoothing the srs
    return SmoothSrs(normalizedsrs, smoothFact)
 

def check_diff(list_1, list_2):
    if len(list_1) != len(list_2):
        return -1
    return len([i for i, j in zip(list_1, list_2) if i != j])


def STC(df,window,fast_lenght,slow_length,smoothing_f):
    macd_diff = MacD(df, fast_lenght, slow_length)
    norm_macd = NormalizeSmoothSrs(df, macd_diff, window,smoothing_f)
    final_stc = NormalizeSmoothSrs(df, norm_macd, window, smoothing_f)
    df['STC'] = final_stc
    return df['STC']


class our_strategy(bs):
    def __init__(self, fast_ema:int, slow_ema:int, atr_length_sell:int, atr_length_buy:int, key_value:int, somoothing_f:float, window:int = 5):
        self.fast_length = fast_ema
        self.slow_length = slow_ema
        self.atr_length_buy = atr_length_buy
        self.atr_length_sell = atr_length_sell
        self.key_value = key_value
        self.somoothing_f = somoothing_f
        self.sl_rate = None
        self.tp_rate = None
        self.window = window

    def calc_signal(self, data: pd.DataFrame) -> pd.Series:
        trailing_stop_sell = UTBot(data, self.key_value, self.atr_length_sell)
        trailing_stop_buy = UTBot(data, self.key_value, self.atr_length_buy)
        close = data['close'].tolist()
        buy_signals = buy_signal(trailing_stop_buy,close)
        sell_signals = sell_signal(trailing_stop_sell,close)
        stc = STC(data,self.window ,self.fast_length, self.slow_length, self.somoothing_f).tolist()
        data['strategy_signal'] = 0
        for i in range(1,len(data)):
            if buy_signals[i] and stc[i] < 25: 
                data.loc[i, 'strategy_signal'] = StrategySignal.ENTER_LONG
            elif sell_signals[i] and stc[i] > 75:
                data.loc[i, 'strategy_signal'] = StrategySignal.CLOSE_LONG

        return data


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


def main():
    # Getting the data and storing it in a csv file 
    # (Note: This will take a while to run! so we run it once and save the data in a csv file)
    path = '' # Enter the path of the file you want to save the data in
    symbol = 'BTCUSDT'
    interval = '30m'
    start_date = int(datetime(year=2023, month=1, day=1).timestamp() * 1000)
    end_date = int(datetime(year=2024, month=1, day=1).timestamp() * 1000)
    get_csv_data(symbol, interval, start_date, end_date,path) # This will take a while to run!

    df = pd.read_csv(path)

    # Creating the strategy instance
    strategy = our_strategy(fast_ema=12,slow_ema=26,atr_length_buy=300,atr_length_sell=1,key_value=1,somoothing_f=0.5)
    balance = 10000
    df['sell'] = None
    df['buy'] = None
    ans = backtest(starting_balance=balance,strategy=strategy,data=df)
    evaluate_strategy(ans,'out_strategy')
    df.to_csv('data_for_part_2.csv', index=False)  # Change 'data.csv' to your desired file name
    plot_strategy(df)
    plot_profit(df)
    

if __name__ == "__main__":
    main()
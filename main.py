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

    # coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT',
    #          'ADAUSDT', 'XRPUSDT', 'DOGEUSDT',
    #          'DOTUSDT', 'UNIUSDT', 'LINKUSDT',
    #          'LTCUSDT', 'BCHUSDT', 'SOLUSDT',
    #          'MATICUSDT', 'XLMUSDT', 'ETCUSDT',
    #          'THETAUSDT', 'VETUSDT', 'TRXUSDT',
    #          'EOSUSDT', 'FILUSDT', 'AAVEUSDT',
    #          'XTZUSDT', 'ATOMUSDT', 'NEOUSDT',
    #          'ALGOUSDT', 'MKRUSDT', 'COMPUSDT',
    #          'KSMUSDT']

def main():
    
    # we can also use the coins 'CROUSDT', 'HTUSDT','CHZUSDT', 'SNXUSDT', 'YFI' if we want to add more coins
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
    
    dtype={
    'open_time': 'datetime64[ms, Asia/Jerusalem]',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
    'close_time': 'datetime64[ms, Asia/Jerusalem]',
    }
    

    # Todo list:
    # 1. To get all the data from the binance API to a csv file
    # 2. To clean the data
    # 3. To create the model that predicts 
    # 4. To create the strategy
    # 5. To backtest the strategy

        
if __name__ == "__main__":
    main()
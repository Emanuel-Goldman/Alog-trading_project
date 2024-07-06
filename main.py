import matplotlib
import pandas as pd
import os
import requests
import numpy as np
from datetime import datetime
from enum import Enum
from typing import List
import matplotlib.pyplot as plt
from strategies import our_strategy 
from models import *
from backtesting import *
from evaluation import *
from get_data import *


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
    plt.plot(df['portfolio_value'], label = 'Portfolio')
    plt.legend()
    plt.show()
    
    
    



def main():
    data = pd.read_csv(r'data\all data.csv')
    strategy = our_strategy(sl_pct = 0.02, tp_pct = 0.05, path = r'data\all data.csv')
    backteting_results = backtest(data, strategy, starting_balance = 1000)
    
    backteting_results.to_csv(r'result\backtesting_results.csv')
    
    # plot_strategy(backteting_results)
    plot_profit(backteting_results)
    

    

if __name__ == "__main__":
    main()
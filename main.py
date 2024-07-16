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
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace = True)
    plt.figure(figsize=(25,10))
    plt.plot(df['portfolio_value'])
    plt.xlabel('Date',fontsize = 15)
    plt.ylabel('USD', fontsize = 15)
    plt.title('Portfolio Value', fontsize = 25)
    # plt.legend()
    plt.show()
    
def plot_profit_vs_hodl_2(df):
    df['test'] = np.where(df['qty'] != 0, df['close'], 0)
    # df = df.iloc[2000:2200]
    df['test'].plot()
    df['close'].plot()
    plt.show()
    
def plot_profit_vs_hodl_3(df):
    
    # Creating an empty column 'test' first
    df['test'] = None
 
# Using a for loop to assign values based on the condition
    for index, row in df.iterrows():
        if index > 0:
            if df.at[index-1, 'qty'] != 0:
                df.at[index, 'test'] = row['close']
    # df['test'] = np.where(df['qty'] != 0, df['close'], None)
    
    df = df.iloc[2200:2500]
    
    df['buy_1'] = np.where(df['buy'] == -2.0, df['close'], None)
    df['sell_1'] = np.where(df['sell'] == 2.0, df['close'], None)
   
    plt.figure(figsize=(25, 10))
    plt.scatter(df.index, df['buy_1'], label = 'Buy',alpha = 1, marker = '^', color = 'green')
    plt.scatter(df.index, df['sell_1'], label = 'Sell', alpha = 1, marker = 'v', color = 'red')
    df['close'].plot(label='Close', lw=2)  # 'lw' is line width
    plt.plot(df.index, df['test'], color='red', label='In Position')  # Scatter plot
    plt.legend()
    plt.show()
 
    
def check_backtesting(df):
    def check_return(df):
        df['return'] = df['close'].pct_change()
        df_in_pos = df[df['balance'] < 1]
        df_in_pos_2 = df[df['qty'] > 1]
        print(f'Test 1')
        print(f'{df_in_pos.shape[0]}=={df_in_pos_2.shape[0]}')
        print(f'the sum of returns when in position = {df_in_pos_2['return'].sum()}')
        print(f'the sum of returns in total = {df['return'].sum()}')
        print(f'the describe of return in position {df_in_pos['return'].describe()}')
        print(f'the describe of return in total {df['return'].describe()}')
        print(f'the return of the coin was {df.iloc[-1]['close']/df.iloc[0]['close']}')
        print(f'the portfolio return is {df.iloc[-1]['portfolio_value']/df.iloc[0]['portfolio_value']}')
    def check_portfolio_val(df):
        df['port_ret'] = df['portfolio_value'].pct_change()
        print(f'The rturns of the portfolio describe {df['port_ret'].describe()}')
        
    # for index, row in df.iterrows():
        

    # check_portfolio_val(df)
    check_return(df)
    # check_portfolio_val(df)


def main():
    data = pd.read_csv(r'data\all data_v3.csv')
    strategy = our_strategy(sl_pct = 0.02, tp_pct = 0.05, path = r'data\all data.csv', commission = 0.0045, )
    backteting_results = backtest(data, strategy, starting_balance = 1000, slippage_factor=10, commission=0.0045)
    
    backteting_results.to_csv(r'result\backtesting_results.csv')
    df = pd.read_csv(r'result\backtesting_results.csv')
    
    
    # check_backtesting(df)
    
    

    # plot_profit(backteting_results)
    
    plot_profit_vs_hodl_3(df)
    
    # plot_strategy(df)
    
    # df = get_binance_historical_data('CHZUSDT', '2h', '2021-01-01')
    # print(df.head())
    # df.to_csv('data/CHZUSDT.csv')
    # df = pd.read_csv('data\CHZUSDT.csv')
    # df.set_index('open_time', inplace = True)
    # df.to_csv('data/CHZUSDT_v2.csv') 
    # df = pd.read_csv('data\period 0.csv')
    # df.set_index('open_time', inplace = True)
    # df.to_csv('data/period 0_v2.csv') 
    # df_pr = pd.read_csv(r'data\all data.csv')
    # df_cl = pd.read_csv('data\CHZUSDT_v2.csv')
    # print(df_pr.head())
    # print('---------------------------------')
    # print(df_cl.head())
    # df_cl.set_index('open_time', inplace = True)
    # df_pr.set_index('open_time', inplace = True)
    # df_pr['open'] = df_cl['open']
    # print(df_pr.head())
    # df_pr.to_csv(r'data\all data_v3.csv')
    

    

if __name__ == "__main__":
    main()
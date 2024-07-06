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
    plt.plot(df['portfolio_value'].pct_change(1)*100, label = 'Portfolio')
    plt.legend()
    plt.show()



def main():

    print("Downloading Data...")
    

if __name__ == "__main__":
    main()
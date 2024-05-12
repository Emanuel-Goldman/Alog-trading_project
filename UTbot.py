import pandas as pd
from typing import List

def TR(high, low, prev_close):
	 return max(high, prev_close) - min(low, prev_close)

def ATR(tr, atr_length):
    atr = [0] * len(tr)
    atr[atr_length - 1] = sum(tr[:atr_length]) / atr_length
    for i in range(atr_length, len(tr)):
        atr[i] = (atr[i - 1] * (atr_length - 1) + tr[i]) / atr_length
    return atr 

def cross(a,b,prev_a,prev_b):
    return a > b and prev_a < prev_b

def buy_signal(trailing_stop:List[int],close:List[int])->List[bool]:

    buy_signal = [False]*len(trailing_stop)
    for i in range(1,len(trailing_stop)):
        buy_signal[i] = cross(a=close[i],b=trailing_stop[i],prev_a=close[i-1],prev_b=trailing_stop[i-1])
    
    return buy_signal


def sell_signal(trailing_stop:List[int],close:List[int])->List[bool]:

    sell_signal = [False]*len(trailing_stop)
    for i in range(1,len(trailing_stop)):
        sell_signal[i] = cross(a=trailing_stop[i],b=close[i],prev_a=trailing_stop[i-1],prev_b=close[i-1])
    
    return sell_signal

def UTBot(data, key_value: int, atr_length: int) -> List[int]:
    LEN = len(data)

    tr:List[int] = [0] * LEN  
    tr[0] = data.iloc[0]['high']-data.iloc[0]['low'] #Taking care for the case there is no close_prev.
    
    for i in range(1,LEN):
        prev_close = data.iloc[i-1]['close']
        high = data.iloc[i]['high']
        low = data.iloc[i]['low']
        tr[i] = TR(high,low,prev_close)


    atr = ATR(tr,atr_length)
    
    loss_threshold = atr * key_value
    
    trailing_stop:List[int] = [0]*LEN
    for i in range(1, LEN):

        prev_close:int = data.iloc[i-1]['close']
        close:int = data.iloc[i]['close']
        if close > trailing_stop[i-1] and prev_close > trailing_stop[i-1]:
            trailing_stop[i] = max(trailing_stop[i-1], close - loss_threshold[i])
        elif close < trailing_stop[i-1] and prev_close < trailing_stop[i-1]:
            trailing_stop[i] = min(trailing_stop[i-1], close + loss_threshold[i])
        elif close > trailing_stop[i-1]:
            trailing_stop[i] = close - loss_threshold[i]
        else:
            trailing_stop[i] = close + loss_threshold[i]
    
    return trailing_stop
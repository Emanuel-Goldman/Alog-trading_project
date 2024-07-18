import pandas as pd
import numpy as np

from models import ActionType, PositionType, Position, StrategySignal
from strategies import BaseStrategy
from evaluation import evaluate_strategy

VISUALIZE = 2         
    
def calc_realistic_price(row: pd.Series ,action_type: ActionType, slippage_factor=1000):
    slippage_rate = ((row['close'] - row['open']) / row['open']) / slippage_factor
    if action_type == ActionType.BUY:
        return row['open'] + row['open'] * slippage_rate
    else:
        return row['open'] - row['open'] * slippage_rate
    
     

def backtest(data: pd.DataFrame, strategy: BaseStrategy, starting_balance: int, slippage_factor: float=1000, commission: float=0.0) -> pd.DataFrame:       
    
    def enter_position(data: pd.DataFrame, index: int,row, curr_qty: float, curr_balance: float, position_type: PositionType) -> Position:
        print('We are in enter_position')
        if position_type == PositionType.LONG:
            buy_price = calc_realistic_price(row, ActionType.BUY, slippage_factor=slippage_factor)
            qty_to_buy = strategy.calc_qty(buy_price, curr_balance, ActionType.BUY)
            position = Position(qty_to_buy, buy_price, position_type)
            data.loc[index, 'qty'] = curr_qty + qty_to_buy
            # data.loc[index, 'balance'] = curr_balance - qty_to_buy * buy_price 
            data.loc[index, 'balance'] = 0
            
        return position
    
    def close_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position: Position, commission: float=0.0045):
        if position.type == PositionType.LONG:
            sell_price = calc_realistic_price(row, ActionType.SELL, slippage_factor=slippage_factor)
            # data.loc[index, 'qty'] = curr_qty - position.qty
            data.loc[index, 'qty'] = 0
            data.loc[index, 'balance'] = curr_balance + (position.qty * (1-commission)) * sell_price 

        
    
    # initialize df 
    data['qty'] = 0.0
    data['balance'] = 0.0
    data['buy'] = np.nan
    data['sell'] = np.nan

    # Calculate strategy signal
    strategy.calc_signal(data)
    
    # Loop through the data to calculate portfolio value
    position: Position = None
    data.reset_index(inplace=True)
    ### ------------------------------------ ###
                    # WRONGLY CALCULATED
    num_trading_days = data.shape[0]
                    # WRONGLY CALCULATED
    ### ------------------------------------ ###
    
    
    for index, row in data.iterrows():
        prev_row = data.iloc[index - 1] if index > 0 else None
        curr_qty = data.loc[index - 1, 'qty'] if index > 0 else 0
        curr_balance = data.loc[index - 1, 'balance'] if index > 0 else starting_balance

        # handle stop loss and take profit
        if position is not None and data.loc[index, 'strategy_signal'] == 0:
            sl_tp_res = strategy.check_sl_tp(data.iloc[index - 1], position)
            if sl_tp_res is not None:
                sl_tp_qty, sl_tp_price, sl_tp_action = sl_tp_res
                    
                if sl_tp_action == ActionType.SELL:
                    curr_balance = curr_balance + sl_tp_qty * sl_tp_price - commission
                    curr_qty = curr_qty - sl_tp_qty
                    print('We are here index is: ', index)
                    data.loc[index,'sell'] = VISUALIZE
                    position = None
        
        # Close position at end of trade
        # if index + 1 >= num_trading_days and position is not None: 
        #     close_position(data, index, row, curr_qty, curr_balance, position)
        #     data.loc[index, 'sell'] = data.loc[index, 'close'] + VISUALIZE
        # Handle enter long signal
        
        if row['strategy_signal'] == 1 and position is None:
            position = enter_position(data, index, row, curr_qty, curr_balance, PositionType.LONG)
            data.loc[index, 'buy'] = - VISUALIZE


        else:
            data.loc[index, 'qty'] = curr_qty
            data.loc[index, 'balance'] = curr_balance
            
        print(index)
        print(row['strategy_signal'])
        
    
    # Calculate portfolio value
    data['portfolio_value'] = data['close'] * data['qty'] + data['balance']
    return data




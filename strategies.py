import pandas as pd
from models import ActionType, Position, PositionType, StrategySignal
from abc import ABC, abstractmethod
from typing import Tuple

class BaseStrategy(ABC):
    def __init__(self, sl_rate: None, tp_rate: None) -> None:
        super().__init__()
        self.sl_rate = sl_rate
        self.tp_rate = tp_rate
    
    @abstractmethod
    def calc_signal(self, data: pd.DataFrame):
        pass

    def calc_qty(self, real_price: float, balance: float, action: ActionType, **kwargs) -> float:
        if action == ActionType.BUY:
            qty = balance / real_price
        
        elif action == ActionType.SELL:
            qty =  balance / real_price
        
        return qty    
    
    def check_sl_tp(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        sl_res = self.is_stop_loss(row, position)
        if sl_res is not None:
            return sl_res
        
        tp_res = self.is_take_profit(row, position)
        if tp_res is not None:
            return tp_res
        
        return None
    
    def is_stop_loss(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the stop-loss level.
        
        Returns:
            Tuple[float, float, ActionType] or None: If stop-loss is triggered, returns a tuple containing quantity and stop-loss price and action type, otherwise returns None.
        """
        if self.sl_rate is not None:
            long_stop_loss_price = position.price * (1 - self.sl_rate)
            if position.type == PositionType.LONG and row['low'] <= long_stop_loss_price:
                return position.qty, long_stop_loss_price, ActionType.SELL

    
    def is_take_profit(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the take-profit level.

        Returns:
            Tuple[float, float, ActionType] or None: If take-profit is triggered, returns a tuple containing quantity and take-profit price and action type, otherwise returns None.
        """
        if self.tp_rate is not None:
            long_take_profit_price = position.price * (1 + self.tp_rate)
            if position.type == PositionType.LONG and row['high'] >= long_take_profit_price:
                return position.qty, long_take_profit_price, ActionType.SELL
            
class our_strategy(BaseStrategy):
    def __init__(self, sl_pct:int, tp_pct:int, path:str):
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.path = path
        
    def calc_signal(self, data: pd.DataFrame):
        df_model_res = pd.read_csv(self.path)
        data['strategy_signal'] = df_model_res['y_pred']
        
    def check_sl_tp(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        
        ### -------------------------------------------------------- ###
        # WE WOULD LIKE TO CREATE A TRAILING STOP LOSS AND TAKE PROFIT
        ### -------------------------------------------------------- ###
        
        def is_stop_loss(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
            if row['close'] <= position.price * (1 - self.sl_pct):
                return position.qty, position.price * (1 - self.sl_pct), ActionType.SELL
            else:
                return None
        
        def is_take_profit(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
            if row['close'] >= position.price * (1 + self.tp_pct):
                return position.qty, position.price * (1 + self.tp_pct), ActionType.SELL
            else:
                return None
        
        sl_res = is_stop_loss(self, row, position)
        if sl_res is not None:
            return sl_res
        
        tp_res = is_take_profit(self,row,position)
        if tp_res is not None:
            return tp_res
        
        else:
            return None
 



        

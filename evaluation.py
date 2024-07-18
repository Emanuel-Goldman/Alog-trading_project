import pandas as pd
import numpy as np

def calc_total_return(portfolio_values):
    # Ensure the portfolio_values is not empty
    if len(portfolio_values) == 0:
        return 0.0
    
    # Calculate total return
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0
    
    return total_return

def calc_annualized_return(portfolio_values):
    trading_days_per_year = 365  # Typically, there are 252 trading days in a year
    total_trading_days = portfolio_values.shape[0]/12
    print(f'total_trading_days = {total_trading_days}')
    total_years = total_trading_days / trading_days_per_year
    print(f'total_years = {total_years}')
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0]
    annualized_return = (total_return ** (1 / total_years)) - 1
    
    return annualized_return

def calc_annualized_sharpe(portfolio_values: pd.Series, rf: float=0.045):
    yearly_trading_days = 365
    annualized_return = calc_annualized_return(portfolio_values)
    annualized_std = portfolio_values.pct_change().std() * np.sqrt(yearly_trading_days)
    print(f'annualized_std = {annualized_std}')
    print(f'annualized_return = {annualized_return}')
    if annualized_std is None or annualized_std == 0:
        return 0
    sharpe = (annualized_return - rf) / annualized_std
    return sharpe

def calc_downside_deviation(portfolio_values):
    porfolio_returns = portfolio_values.pct_change().dropna()
    return porfolio_returns[porfolio_returns < 0].std()

def calc_sortino(portfolio_values, rf=0.0):
    yearly_trading_days = 365
    down_deviation = calc_downside_deviation(portfolio_values) * np.sqrt(yearly_trading_days)
    annualized_return = calc_annualized_return(portfolio_values)
    if down_deviation is None or down_deviation == 0:
        return 0
    sortino = (annualized_return - rf) / down_deviation
    return sortino

def calc_max_drawdown(portfolio_values):
    cumulative_max = portfolio_values.cummax()
    drawdown = (cumulative_max - portfolio_values) / cumulative_max
    return drawdown.max()

def calc_calmar(portfolio_values):
    max_drawdown = calc_max_drawdown(portfolio_values)
    annualized_return = calc_annualized_return(portfolio_values)
    return annualized_return / max_drawdown

def calculate_sharpe_ratio_1(df):
    mean_return = df.iloc[-1]['portfolio_value'] / df.iloc[0]['portfolio_value'] 
    risk_free_rate = 0.0423
    std_return =(df['portfolio_value'].pct_change().std())
    # print(f'The mean return is {mean_return}')
    # print(f'The risk free rate is {risk_free_rate}')
    # print(f'The std return is {std_return}')
    # std_return = sqr(df['portfolio_value'].std())
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio
   

def evaluate_strategy(b_df, strat_name):
    total_return = calc_total_return(b_df['portfolio_value'])
    annualized_return = calc_annualized_return(b_df['portfolio_value'])
    annualized_sharpe = calc_annualized_sharpe(b_df['portfolio_value'])
    sortino_ratio = calc_sortino(b_df['portfolio_value'])
    max_drawdown = calc_max_drawdown(b_df['portfolio_value'])
    calmar_ratio = calc_calmar(b_df['portfolio_value'])
  
    print(f"Results for {strat_name}:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Sharpe Ratio: {annualized_sharpe:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
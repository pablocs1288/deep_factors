import pandas as pd
import numpy as np




def get_portfolio_sharpe_ratio(cumulative_returns, returns, risk_free_rate = 0.0):
    std_ = np.std(returns)
    ret_init = cumulative_returns.head(1).values[0]
    ret_final = cumulative_returns.tail(1).values[0]
    portfolio_return_ = (ret_final - ret_init) / ret_init 
    
    sharpe_ratio = (portfolio_return_ - risk_free_rate) / std_
    
    return sharpe_ratio
    
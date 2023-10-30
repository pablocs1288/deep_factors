import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


def get_portfolio_returns(returns, weights, is_log=True):
    if is_log:
        simple_returns = returns.apply(lambda row: np.dot(row, np.transpose(weights)), axis = 1)
        log_returns = simple_returns.apply(lambda sr: np.log(sr+1))
        return log_returns
    return returns.apply(lambda row: np.dot(row, np.transpose(weights)), axis = 1)


def get_cumulative_returns_over_time(returns, weights):
    return (((1+returns).cumprod(axis=0))-1).dot(weights)


# correct log-returns
def get_cummulative_returns(returns, weights):
    cumulative_returns = get_cumulative_returns_over_time(returns, weights)
    return cumulative_returns


def plot_portfolio_performance(cumulative_returns, weights, tickers, title='Portoflio Performance'):
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

    axs[0].plot(range(len(cumulative_returns)), cumulative_returns, 'black','-')
    axs[0].set_title('Portfolio Performance')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Cumulatice return')

    axs[1].bar(tickers, weights)
    axs[1].set_xlabel('Stock')
    axs[1].set_title('Portfolio Weights')
    axs[1].set_ylabel('Weights')

    fig.suptitle(title, fontsize=16)
import os
import pandas as pd
import numpy as np
import datetime as dt

import yfinance as yahooFinance



def download_data(start, end, tickers):
    prices = pd.DataFrame()

    for ticker in tickers:
        try:
            print('processing ticker {}'.format(ticker))
            # prices[ticker] = data.DataReader(ticker,'yahoo', start, end).loc[:,'Close'] #S&P 500
            prices[ticker] = yahooFinance.download(ticker, start , end).loc[:,'Close'] #S&P 500
        except:
            print('problem with ticker {}'.format(ticker))
            continue
    prices.index = [x.strftime('%Y-%m-%d') for x in prices.index]
    return prices


def reading_data_from_separate_files(root_path):
    df_portfolio_returns = None
    column_names = []
    for file_name in os.listdir(root_path):
        if '.csv' not in file_name:
            continue
        # mapping asset names
        column_names.append(file_name.replace('.csv', ''))
        df_temp = pd.read_csv(root_path+file_name)
        # 1. filling nan
        _fill_nan_with_d_minus_1(df_temp)
        # 2. returns - are not log as markowitz theory does not use log returns
        #_log_returns(df_temp) # mutable object
        _normal_returns(df_temp)
        df_temp = df_temp[['Date', 'linear_returns']]
        if df_portfolio_returns is None:
            df_portfolio_returns = df_temp.copy()
            continue
        # after filling the data this should not be a problem
        df_portfolio_returns = df_portfolio_returns.merge(df_temp, left_on = 'Date', right_on = 'Date', how = 'inner') 
    df_portfolio_returns = df_portfolio_returns.drop('Date', axis = 1)[1:]
    df_portfolio_returns.columns = column_names
    return df_portfolio_returns
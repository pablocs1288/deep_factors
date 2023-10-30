import pandas as pd
import numpy as np
import datetime as dt
from sklearn import preprocessing



def simple_to_log_returns(simple_returns):
    return simple_returns.apply(lambda sr: np.log(sr+1))

def log_to_simple_returns(log_returns):
    _log_returns = log_returns.copy()
    return np.exp(_log_returns) - 1

# The ones the markowitz original theory regards
def get_simple_returns_from_prices(prices):
    prices_ = prices.copy()
    return (prices_ - prices_.shift(1))/prices_.shift(1)



def normalize_data(df):
    x = df.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_n = pd.DataFrame(x_scaled, index=df.index, columns=df.columns)
    return df_n

## train-test sets

def splitting_dataset_by_window(df, date, days_window_train = 252, days_window_test = 30):
    
    # this code avoids null key value
    if isinstance(date, str):
        start_date = dt.datetime.strptime(date, '%Y-%m-%d')
        range_search = start_date - dt.timedelta(days=4)
        
        start_date = start_date.strftime('%Y-%m-%d')
        range_search = range_search.strftime('%Y-%m-%d')
    
    real_starting_date = df.loc[range_search:start_date].iloc[-1].name
    
    df['date'] = df.index
    df_train = df.loc[df['date'] <= real_starting_date].tail(days_window_train)
    df_test = df.loc[df['date'] > real_starting_date].head(days_window_test)
    
    return df_train.drop('date', axis = 1), df_test.drop('date', axis = 1)



# Ammending returns target!!
## this is a bad criterion - it only seek for shocks!
def set_target_returns_extreme_effects_by_percentile(
        y_returns, 
        quantile = .01, 
        rate_target = None , 
        multiplier_correction = 0.8
    ):
    
    y_train_amended = y_returns.copy()
    
    treshold = y_returns.quantile(quantile)
    if rate_target is None:
        rate_target = treshold + (np.abs(treshold)*0.8)
    
    y_train_amended[y_train_amended < treshold] = rate_target
        
    return y_train_amended

## new criterion: set a target 
#fazer uma curva, se quero outperform ( exceso. de 1%) diliu essa porcentagem no tamanho da minha mostra de treino, e fazer com um grid de 1%, 2%, 3%,...... x -> grid; y -> sharpe (para cada iteração!) 

#fazer curva do sharpe em função da curva que estou procurando!
# a metrica boa é usar o sharpe e a serie

def set_target_spreaded_rate(y_returns, rate_target = None):
    """
    Added annualy
    """
    if rate_target is None:
        rate_target = 0.05
    add_each_day = rate_target/252 
    ammended_returns = y_returns + add_each_day
    return ammended_returns

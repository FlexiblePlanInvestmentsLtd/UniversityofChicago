import pandas as pd
import os
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import seaborn as sns
from functools import reduce


def load_data(folder_name, file_name):
     # Load data
    data_path = os.path.dirname(os.getcwd()) + '/' + folder_name
    data_file = data_path + '/' + file_name
    return pd.read_excel(data_file, sheet_name='Universe of broad assets', index_col=0, parse_dates=True)

def comp_ret(series):
        return reduce(lambda x, y: (1 + x) * (1 + y) - 1, series)

def weekly_returns(data):
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data['weekday'] = data.index.day_name()
    
    data['trading_day'] = np.where((data.weekday == 'Tuesday'), 1, 0)   # Start of return calc periods are typically Tuesday
    data['trading_day'] = np.where((data.weekday == 'Wednesday') & 
                                        (data.weekday.shift(1) == 'Monday'), 
                                        1, data.trading_day) # This filter sets Wednesday as the first trading day when Tuesday is a holiday
    data['trading_day'] = np.where((data.weekday == 'Wednesday') & 
                                        (data.weekday.shift(1) == 'Friday'), 
                                        1, data.trading_day) # There are a few cases where markets were shut down both Monday and Tuesday
    # The 9/11 Terrorist attacks were on a Tuesday and shut down markets for a week. Markets re-opened on 9/17, so assuming we place trades 
    # that day and calculate returns through the following EOD Monday. This is the longest holding window in the dataset.
    # TODO: Question, should we just drop 9/17/2001 from the dataset?
    data['trading_day'] = np.where(data.index == '2001-09-17', 1, data.trading_day)
    data['trading_day'] = np.where(data.index == '2001-09-18', 0, data.trading_day)
    data['week'] = data.trading_day.cumsum()
    data.drop(columns=['weekday', 'trading_day'], inplace=True)

    grouped = data.groupby(['week']).agg(comp_ret)
    grouped_dates = data.reset_index().loc[:, ['Date', 'week']]
    grouped_dates['Date'] = grouped_dates['Date'].shift(1)
    grouped_dates = grouped_dates.groupby(['week']).max()
    weekly_returns = pd.merge(left=grouped_dates, right=grouped, right_index=True, left_index=True)

    return weekly_returns.dropna()

def calculate_roc(df, return_columns, period):
    """
    Calculate ROC for each asset based on cumulative returns over a specified period.
    
    :param df: DataFrame containing return data
    :param return_columns: List of columns representing asset returns
    :param period: The number of periods to calculate cumulative returns (ROC)
    :return: DataFrame with ROC for each asset
    """
    roc_df = pd.DataFrame(index=df.index)
    
    for col in return_columns:
        # Calculate the cumulative product of (1 + returns) for the given period, then subtract 1
        roc_df[f'ROC_{col}'] = df[col].rolling(window=period).apply(lambda x: (np.prod(1 + x) - 1), raw=False)
    
    return roc_df.dropna()

def calculate_macd(df, return_columns, short_period=12, long_period=26, signal_period=9):
    """
    Calculate MACD for each asset.
    
    :param df: DataFrame containing return data
    :param return_columns: List of columns representing asset returns
    :param short_period: Short-term EMA period (default 12)
    :param long_period: Long-term EMA period (default 26)
    :param signal_period: Signal line EMA period (default 9)
    :return: DataFrame with MACD line, Signal line, and MACD histogram for each asset
    """
    macd_df = pd.DataFrame(index=df.index)
    
    for col in return_columns:
        short_ema = df[col].ewm(span=short_period, adjust=False).mean()
        long_ema = df[col].ewm(span=long_period, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        macd_df[f'MACD_Line_{col}'] = macd_line
        macd_df[f'Signal_Line_{col}'] = signal_line
        macd_df[f'MACD_Histogram_{col}'] = macd_histogram
    
    return macd_df.dropna()
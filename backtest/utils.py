import pandas as pd
from pandas import Timestamp
import datetime
import numpy as np
import math
import os
from typing import Union, List
pd.options.display.float_format = "{:,.4f}".format

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from arch import arch_model
import talib

import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from functools import reduce
import re




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
    # grouped = data.groupby(['week']).apply(lambda x: (np.prod(1 + x) - 1))
    grouped_dates = data.reset_index().loc[:, ['Date', 'week']]
    grouped_dates['Date'] = grouped_dates['Date']
    grouped_dates = grouped_dates.groupby(['week']).max()
    weekly_returns = pd.merge(left=grouped_dates, right=grouped, right_index=True, left_index=True)
    weekly_returns.index = weekly_returns.Date
    weekly_returns.drop(columns='Date', inplace=True)

    return weekly_returns.dropna()

def monthly_returns(data):
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    
    # Identify the first available trading day in each month if the second day is unavailable
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['trading_day'] = data.groupby(['year', 'month']).cumcount() + 1  # Count trading days in each month
    data['trading_start'] = np.where(data['trading_day'] == 2, 1, 0)
    data['trading_start'] = np.where((data['trading_day'] == 1) & (data['trading_start'].shift(-1) == 0), 1, data['trading_start'])
    
    data['month_group'] = data['trading_start'].cumsum()
    data.drop(columns=['month', 'year', 'trading_day', 'trading_start'], inplace=True)
    
    comp_ret = lambda x: np.prod(1 + x) - 1
    monthly_returns = data.groupby(['month_group']).agg(comp_ret)
    
    # Ensure the last available date in each month_group is used as the period end date
    monthly_end_dates = data.reset_index().groupby('month_group')['Date'].max()
    monthly_returns = pd.merge(left=monthly_end_dates, right=monthly_returns, right_index=True, left_index=True)
    monthly_returns.index = monthly_returns.Date
    monthly_returns.drop(columns='Date', inplace=True)
    
    return monthly_returns.dropna()

def apply_monthly_returns(dataframes):
    """
    Applies the monthly_returns function to each dataframe in a dictionary.

    Args:
    dataframes (dict): Dictionary where each value is a dataframe to transform to monthly returns.

    Returns:
    dict: A dictionary with the same keys as the input, but with monthly return dataframes as values.
    """
    monthly_dataframes = {}
    for period_name, period_df in dataframes.items():
        monthly_dataframes[period_name] = monthly_returns(period_df)
    
    return monthly_dataframes

def calculate_roc(data, periods):
    """
    Calculate ROC for each specified period and concatenate them into a single DataFrame.
    
    :param data: DataFrame containing return data for one asset
    :param periods: List of periods for calculating ROC
    :return: DataFrame with ROC indicators for each specified period
    """
    roc_dfs = []
    for period in periods:
        roc_df = data.rolling(window=period).apply(lambda x: np.prod(1 + x) - 1, raw=False)
        roc_df.columns = [f'ROC{period}_{col}' for col in data.columns]
        roc_dfs.append(roc_df)

    return pd.concat(roc_dfs, axis=1)

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    """
    Calculate MACD for each asset based on returns data.
    
    :param data: DataFrame containing return data for one asset
    :param short_period: Short-term EMA period (default 12)
    :param long_period: Long-term EMA period (default 26)
    :param signal_period: Signal line EMA period (default 9)
    :return: DataFrame with MACD line, Signal line, and MACD histogram
    """
    macd_df = pd.DataFrame(index=data.index)
    
    for col in data.columns:
        price_data = (1 + data[col]).cumprod()
        short_ema = price_data.ewm(span=short_period, adjust=False).mean()
        long_ema = price_data.ewm(span=long_period, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line

        macd_df[f'MACD_Line_{col}'] = macd_line
        macd_df[f'Signal_Line_{col}'] = signal_line
        macd_df[f'MACD_Histogram_{col}'] = macd_histogram
    
    return macd_df

def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a pandas DataFrame or Series that contains
    daily return data by first converting it to a cumulative price series.
    
    Parameters:
    data (pd.Series or pd.DataFrame): Daily returns data.
    window (int): The period for calculating RSI, default is 14.
    
    Returns:
    pd.DataFrame: RSI and Stochastic RSI values.
    """
    price_data = (1 + data).cumprod()
    delta = price_data.diff()
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    if isinstance(data, pd.DataFrame):
        gain = pd.DataFrame(gain, index=price_data.index, columns=price_data.columns)
        loss = pd.DataFrame(loss, index=price_data.index, columns=price_data.columns)
    else:
        gain = pd.Series(gain, index=price_data.index)
        loss = pd.Series(loss, index=price_data.index)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    if isinstance(rsi, pd.DataFrame):
        rsi.columns = [f'RSI_{col}' for col in rsi.columns]

    stoch_rsi = (rsi - rsi.rolling(window=window).min()) / (rsi.rolling(window=window).max() - rsi.rolling(window=window).min())

    if isinstance(stoch_rsi, pd.DataFrame):
        stoch_rsi.columns = [f'Stoch_{col}' for col in stoch_rsi.columns]
    
    rsi_df = pd.concat([rsi, stoch_rsi], axis=1)

    return rsi_df.dropna()

def calculate_bollinger_bands(data, windows=[20, 90, 180, 240], num_std=2):
    """
    Compute Bollinger Bands for multiple rolling window sizes based on return data
    by first converting returns to a cumulative price series.
    
    Parameters:
    - data (pd.DataFrame): A DataFrame of daily percent return data.
    - windows (list of int): List of rolling window sizes for moving averages.
    - num_std (int): Number of standard deviations for the bands (default is 2).

    Returns:
    - pd.DataFrame: Data with Bollinger Bands (MA, Upper, Lower) for each specified window.
    """
    price_data = (1 + data).cumprod()
    
    bollinger_dfs = []
    
    for window in windows:
        rolling_mean = price_data.rolling(window=window).mean()
        rolling_std = price_data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        rolling_mean.columns = [f'SMA{window}_{col}' for col in data.columns]
        upper_band.columns = [f'Upper{window}_{col}' for col in data.columns]
        lower_band.columns = [f'Lower{window}_{col}' for col in data.columns]
        
        bollinger_df = pd.concat([rolling_mean, upper_band, lower_band], axis=1)
        bollinger_dfs.append(bollinger_df)
    
    return pd.concat(bollinger_dfs, axis=1).dropna()

def calc_hilbert(returns):
    data = (1 + returns).cumprod()
    ht_transform = pd.DataFrame()

    for asset in returns.columns:
        ht_transform['ht_dcperiod_' + asset] = talib.HT_DCPERIOD(data[asset])                   # Calculate Hilbert Transform - Dominant Cycle Period
        ht_transform['ht_dcphase_' + asset] = talib.HT_DCPHASE(data[asset])                     # Calculate Hilbert Transform - Dominant Cycle Phase
        ht_transform['inphase_' + asset], data['quadrature'] = talib.HT_PHASOR(data[asset])     # Calculate Hilbert Transform - Phasor Components
        ht_transform['sine_' + asset], data['leadsine'] = talib.HT_SINE(data[asset])            # Calculate Hilbert Transform - SineWave
        ht_transform['ht_trendmode_' + asset] = talib.HT_TRENDMODE(data[asset])                 # Calculate Hilbert Transform - Trend vs Cycle Mode

    return ht_transform.dropna()

def calc_ewma_volatility(excess_returns, asset_col, theta=0.94, initial_vol=0.2 / np.sqrt(252)):
    """
    Calculate EWMA volatility for a single asset.

    :param excess_returns: DataFrame containing returns for one asset
    :param asset_col: Name of the asset column
    :param theta: Smoothing parameter for EWMA (default 0.94)
    :param initial_vol: Initial volatility value (default 0.2 / sqrt(252))
    :return: DataFrame with EWMA volatility for the specified asset
    """
    var_t0 = initial_vol ** 2
    ewma_var = [var_t0]
    
    for d in excess_returns.index:
        new_ewma_var = ewma_var[-1] * theta + (excess_returns.loc[d, asset_col] ** 2) * (1 - theta)
        ewma_var.append(new_ewma_var)
    ewma_var.pop(0)  # Remove the initial var_t0
    ewma_vol = [np.sqrt(v) for v in ewma_var]
    
    return pd.DataFrame({f'EWMA_vol_{asset_col}': ewma_vol}, index=excess_returns.index)

def apply_indicators_broad_assets(data, 
                     indicators, 
                     assets, 
                     reference_data, 
                     theta=0.94, 
                     initial_vol=0.2 / np.sqrt(252),
                     rolling_windows=[20, 60, 126, 252, 504],
                     bollinger_windows=[20, 90, 180, 240],
                     lags=[1, 2, 4, 6, 12]):
    """
    Apply indicators, rolling volatilities, expanding skew, and Hilbert Transform indicators for each asset.
    
    :param data: DataFrame with weekly or monthly returns for each asset
    :param indicators: Dictionary of indicator functions
    :param assets: List of asset columns to process
    :param reference_data: DataFrame of daily returns for calculations
    :param theta: Smoothing parameter for EWMA
    :param initial_vol: Initial volatility for EWMA calculation
    :param rolling_windows: List of windows for calculating rolling volatilities
    :param lags: List of lag periods to create lagged variables
    """

    # Calculate indicators (MACD, RSI, Bollinger Bands) using daily data in reference_data
    macd_df = calculate_macd(reference_data, short_period=12, long_period=26, signal_period=9)
    rsi_df = calculate_rsi(reference_data, window=14)
    bollinger_df = calculate_bollinger_bands(reference_data, windows=bollinger_windows)

    # Calculate rolling statistics using daily data
    rolling_vol_dfs = {window: reference_data.rolling(window=window).std() for window in rolling_windows}
    rolling_skew_dfs = {window: reference_data.rolling(window=window).skew() for window in rolling_windows}
    rolling_kurt_dfs = {window: reference_data.rolling(window=window).kurt() for window in rolling_windows}

    # Expanding skew/kurtosis using daily data
    expanding_skew_df = reference_data.expanding(min_periods=(252 * 2)).skew().dropna()
    expanding_kurt_df = reference_data.expanding(min_periods=(252 * 2)).kurt().dropna()

    # Hilbert Transform using daily data
    hilbert_df = calc_hilbert(reference_data)

    # Loop through each asset and calculate indicators
    asset_dataframes = {}
    for asset in assets:
        indicator_dfs = []

        # Get MACD, RSI, and Bollinger Bands indicators for the asset
        macd_asset_df = macd_df[[f'MACD_Line_{asset}', f'Signal_Line_{asset}', f'MACD_Histogram_{asset}']]
        rsi_asset_df = rsi_df[[f'RSI_{asset}', f'Stoch_RSI_{asset}']]
        bollinger_asset_df = bollinger_df[[f'SMA{window}_{asset}' for window in bollinger_windows] + 
                                          [f'Upper{window}_{asset}' for window in bollinger_windows] +
                                          [f'Lower{window}_{asset}' for window in bollinger_windows]]

        # Append indicators
        indicator_dfs.extend([macd_asset_df, rsi_asset_df, bollinger_asset_df])

        # Apply custom indicators from the indicators dictionary
        for indicator_name, indicator_func in indicators.items():
            indicator_dfs.append(indicator_func(data[[asset]]))

        # Calculate EWMA volatility for each asset
        ewma_vol_df = calc_ewma_volatility(reference_data, asset, theta=theta, initial_vol=initial_vol)
        indicator_dfs.append(ewma_vol_df)

        # Add rolling and expanding stats
        for window, df in rolling_vol_dfs.items():
            indicator_dfs.append(df[[asset]].rename(columns={asset: f'RollingVol_{window}_{asset}'}))
        for window, df in rolling_skew_dfs.items():
            indicator_dfs.append(df[[asset]].rename(columns={asset: f'RollingSkew_{window}_{asset}'}))
        for window, df in rolling_kurt_dfs.items():
            indicator_dfs.append(df[[asset]].rename(columns={asset: f'RollingKurt_{window}_{asset}'}))
        indicator_dfs.extend([expanding_skew_df[[asset]].rename(columns={asset: f'ExpandingSkew_{asset}'}),
                              expanding_kurt_df[[asset]].rename(columns={asset: f'ExpandingKurt_{asset}'})])

        # Add Hilbert Transform for each asset
        indicator_dfs.append(hilbert_df[[f'ht_dcperiod_{asset}', f'ht_dcphase_{asset}', f'inphase_{asset}', 
                                         f'sine_{asset}', f'ht_trendmode_{asset}']])

        # Combine indicators and apply lags
        combined_df = pd.concat(indicator_dfs, axis=1).dropna()
        lagged_features = [combined_df.shift(lag).add_suffix(f'_lag{lag}') for lag in lags]
        combined_df = pd.concat([combined_df] + lagged_features, axis=1).dropna()

        # Add one-hot encoding for the asset and binary target variable (optional)
        asset_columns = [col for col in reference_data.columns if col.startswith('Asset')]
        one_hot_df = pd.DataFrame(0, index=combined_df.index, columns=[f'Asset_{i+1}' for i in range(len(asset_columns))])
        one_hot_df[f'Asset_{asset_columns.index(asset) + 1}'] = 1
        combined_df = pd.concat([combined_df, one_hot_df], axis=1)


        combined_df.columns = [re.sub(f'_Asset \d+', '', col) if not col.startswith('Asset_') else col
                               for col in combined_df.columns]      

        asset_dataframes[asset] = combined_df
    
    return asset_dataframes

def apply_indicators_equity_baskets(data, 
                     indicators, 
                     assets, 
                     reference_data, 
                     theta=0.94, 
                     initial_vol=0.2 / np.sqrt(252),
                     rolling_windows=[20, 60, 126, 252, 504],
                     bollinger_windows=[20, 90, 180, 240],
                     lags=[1, 2, 4, 6, 12]):
    """
    Apply indicators, rolling volatilities, expanding skew, and Hilbert Transform indicators for each asset.
    
    :param data: DataFrame with weekly or monthly returns for each asset
    :param indicators: Dictionary of indicator functions
    :param assets: List of asset columns to process
    :param reference_data: DataFrame of daily returns for calculations
    :param theta: Smoothing parameter for EWMA
    :param initial_vol: Initial volatility for EWMA calculation
    :param rolling_windows: List of windows for calculating rolling volatilities
    :param lags: List of lag periods to create lagged variables
    """

    # Calculate indicators (MACD, RSI, Bollinger Bands) using daily data in reference_data
    macd_df = calculate_macd(reference_data, short_period=12, long_period=26, signal_period=9)
    rsi_df = calculate_rsi(reference_data, window=14)
    bollinger_df = calculate_bollinger_bands(reference_data, windows=bollinger_windows)

    # Calculate rolling statistics using daily data
    rolling_vol_dfs = {window: reference_data.rolling(window=window).std() for window in rolling_windows}
    rolling_skew_dfs = {window: reference_data.rolling(window=window).skew() for window in rolling_windows}
    rolling_kurt_dfs = {window: reference_data.rolling(window=window).kurt() for window in rolling_windows}

    # Expanding skew/kurtosis using daily data
    expanding_skew_df = reference_data.expanding(min_periods=(252 * 2)).skew().dropna()
    expanding_kurt_df = reference_data.expanding(min_periods=(252 * 2)).kurt().dropna()

    # Hilbert Transform using daily data
    hilbert_df = calc_hilbert(reference_data)

    # Loop through each asset and calculate indicators
    asset_dataframes = {}
    for asset in assets:
        indicator_dfs = []

        # Get MACD, RSI, and Bollinger Bands indicators for the asset
        macd_asset_df = macd_df[[f'MACD_Line_{asset}', f'Signal_Line_{asset}', f'MACD_Histogram_{asset}']]
        rsi_asset_df = rsi_df[[f'RSI_{asset}', f'Stoch_RSI_{asset}']]
        bollinger_asset_df = bollinger_df[[f'SMA{window}_{asset}' for window in bollinger_windows] + 
                                          [f'Upper{window}_{asset}' for window in bollinger_windows] +
                                          [f'Lower{window}_{asset}' for window in bollinger_windows]]

        # Append indicators
        indicator_dfs.extend([macd_asset_df, rsi_asset_df, bollinger_asset_df])

        # Apply custom indicators from the indicators dictionary
        for indicator_name, indicator_func in indicators.items():
            indicator_dfs.append(indicator_func(data[[asset]]))

        # Calculate EWMA volatility for each asset
        ewma_vol_df = calc_ewma_volatility(reference_data, asset, theta=theta, initial_vol=initial_vol)
        indicator_dfs.append(ewma_vol_df)

        # Add rolling and expanding stats
        for window, df in rolling_vol_dfs.items():
            indicator_dfs.append(df[[asset]].rename(columns={asset: f'RollingVol_{window}_{asset}'}))
        for window, df in rolling_skew_dfs.items():
            indicator_dfs.append(df[[asset]].rename(columns={asset: f'RollingSkew_{window}_{asset}'}))
        for window, df in rolling_kurt_dfs.items():
            indicator_dfs.append(df[[asset]].rename(columns={asset: f'RollingKurt_{window}_{asset}'}))
        indicator_dfs.extend([expanding_skew_df[[asset]].rename(columns={asset: f'ExpandingSkew_{asset}'}),
                              expanding_kurt_df[[asset]].rename(columns={asset: f'ExpandingKurt_{asset}'})])

        # Add Hilbert Transform for each asset
        indicator_dfs.append(hilbert_df[[f'ht_dcperiod_{asset}', f'ht_dcphase_{asset}', f'inphase_{asset}', 
                                         f'sine_{asset}', f'ht_trendmode_{asset}']])

        # Combine indicators and apply lags
        combined_df = pd.concat(indicator_dfs, axis=1).dropna()
        lagged_features = [combined_df.shift(lag).add_suffix(f'_lag{lag}') for lag in lags]
        combined_df = pd.concat([combined_df] + lagged_features, axis=1).dropna()
        combined_df.columns = [re.sub(f'_Asset \d+', '', col) if not col.startswith('Asset_') else col
                               for col in combined_df.columns]      

        asset_dataframes[asset] = combined_df
    
    return asset_dataframes

def calc_summary_statistics(
    returns: Union[pd.DataFrame, List],
    annual_factor: int = None,
    provided_excess_returns: bool = None,
    rf: Union[pd.Series, pd.DataFrame] = None,
    var_quantile: Union[float, List] = .05,
    timeframes: Union[None, dict] = None,
    # return_tangency_weights: bool = True,
    correlations: Union[bool, List] = True,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
    _timeframe_name: str = None,
):
    """
    Calculates summary statistics for a time series of returns.

    Parameters:
    returns (pd.DataFrame or List): Time series of returns.
    annual_factor (int, default=None): Factor for annualizing returns.
    provided_excess_returns (bool, default=None): Whether excess returns are already provided.
    rf (pd.Series or pd.DataFrame, default=None): Risk-free rate data.
    var_quantile (float or list, default=0.05): Quantile for Value at Risk (VaR) calculation.
    timeframes (dict or None, default=None): Dictionary of timeframes to calculate statistics for each period.
    # return_tangency_weights (bool, default=True): If True, returns tangency portfolio weights.
    correlations (bool or list, default=True): If True, returns correlations, or specify columns for correlations.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): Whether to drop specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: Summary statistics of the returns.
    """
    returns = returns.copy()
    if isinstance(rf, (pd.Series, pd.DataFrame)):
        rf = rf.copy()
        if provided_excess_returns is True:
            raise Exception(
                'rf is provided but excess returns were provided as well.'
                'Remove "rf" or set "provided_excess_returns" to None or False'
            )
        
    if isinstance(returns, list):
        returns_list = returns[:]
        returns = pd.DataFrame({})
        for series in returns_list:
            returns = returns.merge(series, right_index=True, left_index=True, how='outer')
    """
    This functions returns the summary statistics for the input total/excess returns passed
    into the function
    """
    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    try:
        returns.index = pd.to_datetime(returns.index.map(lambda x: x.date()))
    except AttributeError:
        print('Could not convert "date" index to datetime.date')
        pass

    returns = returns.apply(lambda x: x.astype(float))

    if annual_factor is None:
        print('Assuming monthly returns with annualization term of 12')
        annual_factor = 12

    if provided_excess_returns is None:
        print(
            'Assuming excess returns were provided to calculate Sharpe.'
            ' If returns were provided (steady of excess returns), the column "Sharpe" is actually "Mean/Volatility"'
        )
        provided_excess_returns = True
    elif provided_excess_returns is False:
        if rf is not None:
            if len(rf.index) != len(returns.index):
                raise Exception('"rf" index must be the same lenght as "returns"')
            print('"rf" is used to subtract returns to calculate Sharpe, but nothing else')

    if isinstance(timeframes, dict):
        all_timeframes_summary_statistics = pd.DataFrame({})
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_returns = returns.loc[timeframe[0]:timeframe[1]]
            elif timeframe[0]:
                timeframe_returns = returns.loc[timeframe[0]:]
            elif timeframe[1]:
                timeframe_returns = returns.loc[:timeframe[1]]
            else:
                timeframe_returns = returns.copy()
            if len(timeframe_returns.index) == 0:
                raise Exception(f'No returns for {name} timeframe')
            timeframe_returns = timeframe_returns.rename(columns=lambda c: c + f' {name}')
            timeframe_summary_statistics = calc_summary_statistics(
                returns=timeframe_returns,
                annual_factor=annual_factor,
                provided_excess_returns=provided_excess_returns,
                rf=rf,
                var_quantile=var_quantile,
                timeframes=None,
                correlations=correlations,
                _timeframe_name=name,
                keep_columns=keep_columns,
                drop_columns=drop_columns,
                keep_indexes=keep_indexes,
                drop_indexes=drop_indexes,
                drop_before_keep=drop_before_keep
            )
            all_timeframes_summary_statistics = pd.concat(
                [all_timeframes_summary_statistics, timeframe_summary_statistics],
                axis=0
            )
        return all_timeframes_summary_statistics

    summary_statistics = pd.DataFrame(index=returns.columns)
    summary_statistics['Mean'] = returns.mean()
    summary_statistics['Annualized Mean'] = returns.mean() * annual_factor
    summary_statistics['Vol'] = returns.std()
    summary_statistics['Annualized Vol'] = returns.std() * np.sqrt(annual_factor)
    try:
        if not provided_excess_returns:
            if type(rf) == pd.DataFrame:
                rf = rf.iloc[:, 0].to_list()
            elif type(rf) == pd.Series:
                rf = rf.to_list()
            else:
                raise Exception('"rf" must be either a pd.DataFrame or pd.Series')
            excess_returns = returns.apply(lambda x: x - rf)
            summary_statistics['Sharpe'] = excess_returns.mean() / returns.std()
        else:
            summary_statistics['Sharpe'] = returns.mean() / returns.std()
    except Exception as e:
        print(f'Could not calculate Sharpe: {e}')
    summary_statistics['Annualized Sharpe'] = summary_statistics['Sharpe'] * np.sqrt(annual_factor)
    summary_statistics['Min'] = returns.min()
    summary_statistics['Max'] = returns.max()
    summary_statistics['Skewness'] = returns.skew()
    summary_statistics['Excess Kurtosis'] = returns.kurtosis()
    var_quantile = [var_quantile] if isinstance(var_quantile, (float, int)) else var_quantile
    for var_q in var_quantile:
        summary_statistics[f'Historical VaR ({var_q:.2%})'] = returns.quantile(var_q, axis = 0)
        summary_statistics[f'Annualized Historical VaR ({var_q:.2%})'] = returns.quantile(var_q, axis = 0) * np.sqrt(annual_factor)
        summary_statistics[f'Historical CVaR ({var_q:.2%})'] = returns[returns <= returns.quantile(var_q, axis = 0)].mean()
        summary_statistics[f'Annualized Historical CVaR ({var_q:.2%})'] = returns[returns <= returns.quantile(var_q, axis = 0)].mean() * np.sqrt(annual_factor)
    
    wealth_index = 1000 * (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    summary_statistics['Max Drawdown'] = drawdowns.min()
    summary_statistics['Peak'] = [previous_peaks[col][:drawdowns[col].idxmin()].idxmax() for col in previous_peaks.columns]
    summary_statistics['Bottom'] = drawdowns.idxmin()

    # if return_tangency_weights:
    #     tangency_weights = calc_tangency_weights(returns)
    #     summary_statistics = summary_statistics.join(tangency_weights)
    
    recovery_date = []
    for col in wealth_index.columns:
        prev_max = previous_peaks[col][:drawdowns[col].idxmin()].max()
        recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin():]]).T
        recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
    summary_statistics['Recovery'] = recovery_date
    try:
        summary_statistics["Duration (days)"] = [
            (i - j).days if i != "-" else "-" for i, j in
            zip(summary_statistics["Recovery"], summary_statistics["Bottom"])
        ]
    except (AttributeError, TypeError) as e:
        print(f'Cannot calculate "Drawdown Duration" calculation because there was no recovery or because index are not dates: {str(e)}')

    if correlations is True or isinstance(correlations, list):
        returns_corr = returns.corr()
        if _timeframe_name:
            returns_corr = returns_corr.rename(columns=lambda c: c.replace(f' {_timeframe_name}', ''))
        returns_corr = returns_corr.rename(columns=lambda c: c + ' Correlation')
        if isinstance(correlations, list):
            correlation_names = [c + ' Correlation' for c  in correlations]
            not_in_returns_corr = [c for c in correlation_names if c not in returns_corr.columns]
            if len(not_in_returns_corr) > 0:
                not_in_returns_corr = ", ".join([c.replace(' Correlation', '') for c in not_in_returns_corr])
                raise Exception(f'{not_in_returns_corr} not in returns columns')
            returns_corr = returns_corr[[c + ' Correlation' for c  in correlations]]
        summary_statistics = summary_statistics.join(returns_corr)
    
    return filter_columns_and_indexes(
        summary_statistics,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep
    )

def filter_columns_and_indexes(
    df: pd.DataFrame,
    keep_columns: Union[list, str],
    drop_columns: Union[list, str],
    keep_indexes: Union[list, str],
    drop_indexes: Union[list, str],
    drop_before_keep: bool = False
):
    """
    Filters a DataFrame based on specified columns and indexes.

    Parameters:
    df (pd.DataFrame): DataFrame to be filtered.
    keep_columns (list or str): Columns to keep in the DataFrame.
    drop_columns (list or str): Columns to drop from the DataFrame.
    keep_indexes (list or str): Indexes to keep in the DataFrame.
    drop_indexes (list or str): Indexes to drop from the DataFrame.
    drop_before_keep (bool, default=False): Whether to drop specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        return df
    df = df.copy()
    # Columns
    if keep_columns is not None:
        keep_columns = "(?i)" + "|".join(keep_columns) if isinstance(keep_columns, list) else "(?i)" + keep_columns
    else:
        keep_columns = None
    if drop_columns is not None:
        drop_columns = "(?i)" + "|".join(drop_columns) if isinstance(drop_columns, list) else "(?i)" + drop_columns
    else:
        drop_columns = None
    if not drop_before_keep:
        if keep_columns is not None:
            df = df.filter(regex=keep_columns)
    if drop_columns is not None:
        df = df.drop(columns=df.filter(regex=drop_columns).columns)
    if drop_before_keep:
        if keep_columns is not None:
            df = df.filter(regex=keep_columns)
    # Indexes
    if keep_indexes is not None:
        keep_indexes = "(?i)" + "|".join(keep_indexes) if isinstance(keep_indexes, list) else "(?i)" + keep_indexes
    else:
        keep_indexes = None
    if drop_indexes is not None:
        drop_indexes = "(?i)" + "|".join(drop_indexes) if isinstance(drop_indexes, list) else "(?i)" + drop_indexes
    else:
        drop_indexes = None
    if not drop_before_keep:
        if keep_indexes is not None:
            df = df.filter(regex=keep_indexes, axis=0)
    if drop_indexes is not None:
        df = df.drop(index=df.filter(regex=drop_indexes, axis=0).index)
    if drop_before_keep:
        if keep_indexes is not None:
            df = df.filter(regex=keep_indexes, axis=0)
    return df

def calc_cumulative_returns(
    returns: Union[pd.DataFrame, pd.Series],
    return_plot: bool = True,
    fig_size: tuple = (7, 5),
    return_series: bool = False,
    name: str = None,
    timeframes: Union[None, dict] = None,
):
    """
    Calculates cumulative returns from a time series of returns.

    Parameters:
    returns (pd.DataFrame or pd.Series): Time series of returns.
    return_plot (bool, default=True): If True, plots the cumulative returns.
    fig_size (tuple, default=(7, 5)): Size of the plot for cumulative returns.
    return_series (bool, default=False): If True, returns the cumulative returns as a DataFrame.
    name (str, default=None): Name for the title of the plot or the cumulative return series.
    timeframes (dict or None, default=None): Dictionary of timeframes to calculate cumulative returns for each period.

    Returns:
    pd.DataFrame or None: Returns cumulative returns DataFrame if `return_series` is True.
    """
    if timeframes is not None:
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_returns = returns.loc[timeframe[0]:timeframe[1]]
            elif timeframe[0]:
                timeframe_returns = returns.loc[timeframe[0]:]
            elif timeframe[1]:
                timeframe_returns = returns.loc[:timeframe[1]]
            else:
                timeframe_returns = returns.copy()
            if len(timeframe_returns.index) == 0:
                raise Exception(f'No returns for {name} timeframe')
            calc_cumulative_returns(
                timeframe_returns,
                return_plot=return_plot,
                fig_size=fig_size,
                return_series=return_series,
                name=name,
                timeframes=None
            )
        return
    returns = returns.copy()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    returns = returns.apply(lambda x: x.astype(float))
    returns = returns.apply(lambda x: x + 1)
    returns = returns.cumprod()
    returns = returns.apply(lambda x: x - 1)
    title = f'Cumulative Returns {name}' if name else 'Cumulative Returns'
    if return_plot:
        returns.plot(
            title=title,
            figsize=fig_size,
            grid=True,
            xlabel='Date',
            ylabel='Cumulative Returns'
        )
    if return_series:
        return returns
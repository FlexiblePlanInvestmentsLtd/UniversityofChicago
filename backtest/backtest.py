import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import minimize

import sys
import os
import logging
import threading
from joblib import Memory, parallel_backend
import joblib
from ray.util.joblib import register_ray
import ray
register_ray()

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)
from utils import *

# TODO: Add a logger for model training details
class MissingHistoricalReturns(Exception):
    def __init__(self):
        super().__init__()

class FeaturesTargetsLengthException(Exception):
    def __init__(self, feature_name, feature_len, target_len):
        message = f'{feature_name} has length {feature_len}, which is different from target length of {target_len}'
        super().__init__(message)


class Backtest:
    # Note: This class could be generalized to a base class that has child classes for backtesting different
    # classes of models
    def __init__(self, model: dict, base_per, update_freq: int):
        """
        Initialize the backtest object with a model, base period, and update frequency.
        The base period is the starting date for the backtest and the update frequency is the number of weeks.

        :param model: A dictionary of the model and dimensionality reduction object to be used in the backtest
        :param base_per: The starting date for the backtest
        :param update_freq: The number of weeks between updates
        """
        self.model = Classifier(**model)
        self.base_per = datetime.datetime.strptime(base_per, '%Y-%m-%d')
        self.assets = {}
        self.target = None
        self.update_freq = update_freq
        self.lookbacks = None
        self.dates_inter = None
        self.returns = None
        self.predictions = None
        self.mvo_weights = None
        self.risk_parity_weights = None
        self.strat_rets = pd.DataFrame()  
        
        # Attempting to speed up MVO with parallelism
        ray.init(address='auto', 
                 logging_level=logging.FATAL, 
                 log_to_driver=False, ignore_reinit_error=True)

    def read_returns(self, data_path, universe_returns):
        self.returns = pd.read_excel(data_path + universe_returns + '.xlsx', index_col=0, parse_dates=True)

    @staticmethod
    def set_targets(target, method=None):
        """
        Provides different methods for calculating the target variables for classifier training.
        :param method: The method for setting the target variable
            - 'global_mean': Set the target to 1 for returns greater than the global mean up to a date
            - 'asset_mean': Set the target to 1 for returns greater than the asset mean up to a date
            - 'global_std': Set the target to 1 for returns greater than the global mean + std up to a date
            - 'asset_std': Set the target to 1 for returns greater than the asset mean + std up to a date
        """
        if method == 'global_mean':
            threshold = target.expanding().mean().mean(axis=1).apply(lambda x: max(x, 0))
            return pd.DataFrame(np.where((target.T > threshold).T, 1, 0), columns=target.columns, index=target.index)
        
        elif method == 'asset_mean':
            threshold = target.expanding().mean().apply(lambda x: [max(i, 0) for i in x])
            return pd.DataFrame(np.where(target > threshold, 1, 0), columns=target.columns, index=target.index)
        
        elif method == 'global_std':
            mean = target.expanding().mean().mean(axis=1)
            std = target.expanding(method='table').apply(lambda x: np.std(x), raw=True, engine='numba').iloc[:, 0]
            threshold = (mean + std).apply(lambda x: max(x, 0))
            return pd.DataFrame(np.where((target.T > threshold).T, 1, 0), columns=target.columns, index=target.index)
        
        elif method == 'asset_std':
            mean = target.expanding().mean().apply(lambda x: [max(i, 0) for i in x])
            std = target.expanding().std()
            return pd.DataFrame(np.where(target > mean + std, 1, 0), columns=target.columns, index=target.index)
       
        else:
            if method not in ('basic', None):
                print('Invalid method for setting target variable. Defaulting to basic method.')
            return pd.DataFrame(np.where(target > 0, 1, 0), columns=target.columns, index=target.index)
    
    def read_data(self, data_path, universe, assets, file_str_func=None, target_threshold=None):
        if file_str_func:
            file_names = file_str_func(universe=universe, assets=assets)
            for asset, file in zip(assets, file_names):
                self.assets[asset] = pd.read_excel(data_path + file + '.xlsx', 
                                                sheet_name=0, index_col=0, parse_dates=True)
        else:
            for table in assets:
                self.assets[table] = pd.read_excel(data_path + table + '.xlsx', 
                                                sheet_name=0, index_col=0, parse_dates=True)
            
        # Logic to set the target variable - Need to filter returns to intersection of dates
        dates_inter = self.assets[assets[0]].index
        for asset in self.assets:
            cand = self.assets[asset].index
            dates_inter = cand if len(cand) > len(dates_inter) else dates_inter

        if self.returns.index[-1] == dates_inter[-1]:
            self.assets = {key:value.iloc[:-1] for key, value in self.assets.items()}
            dates_inter = dates_inter[:-1]
        self.dates_inter = dates_inter

        target = self.returns.shift(-1).dropna()
        target = target.loc[self.dates_inter]
        self.target = Backtest.set_targets(target=target, method=target_threshold)

    def build_training_dataset(self, training_per):
        features_df = pd.concat([asset.loc[:training_per] for asset in self.assets.values()], axis=0)
        features_df = features_df.reset_index(drop=True)
        
        # This simple method of melting the target variable breaks down when the assets don't have the same length
        # target_df = self.target.loc[:training_per]
        # target_df = target_df.melt(value_name='target').drop(columns='variable')
        
        # Testing more robust method of building the target variable
        target_df = pd.DataFrame()
        for asset, frame in self.assets.items():
            asset_dates = frame.loc[:training_per].index
            target_df = pd.concat([target_df, self.target[asset].loc[asset_dates]], axis=0)

        target_df = target_df.melt(value_name='target').dropna().drop(columns='variable')
        
        return features_df, target_df
    
    def compute_lookbacks(self, update_freq=None, data_freq='weekly'):
        if self.returns is None:
            raise MissingHistoricalReturns
        
        if update_freq: # This should be considered an override of the update_freq that is passed for initialization
            self.update_freq = update_freq
        
        # If frequency doesn't perfectly divide the length of historical
        # returns, we should extend the self.base_per attribute
        train_pers = self.returns.loc[self.base_per:]
        if (resid := len(train_pers) % self.update_freq) != 0:
            self.base_per += timedelta(weeks=resid)
            train_pers = self.returns.loc[self.base_per:]

        # Logic to extract start and end dates of periods based on historical returns
        # data and the frequency of updates. 
        lookbacks_count = len(train_pers) // self.update_freq
        if data_freq == 'weekly':
            lookbacks = [self.base_per + (timedelta(weeks=i) * self.update_freq) for i in range(lookbacks_count)]
        else:
            lookbacks = [self.base_per + (relativedelta(months=i) * self.update_freq) for i in range(lookbacks_count)]
        self.lookbacks = lookbacks

    def train_model(self, features, target, param_grid, dump_version=None):
        self.model.train(param_grid=param_grid, features=features, target=target)

        if dump_version:
            joblib.dump(self.model.pipeline, dump_version + '.joblib')

    def generate_preds(self, data):
        return self.model.predict(data)

    def record_predictions(self, assets, param_grid=None):
        # Loop: 
        self.predictions = pd.DataFrame(columns=assets)
        for i in range(len(self.lookbacks)):
            # L.1. Train the model on data up until a lookback period
            features, target = self.build_training_dataset(self.lookbacks[i])
            self.train_model(features=features, target=target, 
                             param_grid=param_grid)#, dump_version=f'mod_backtest_{self.lookbacks[i]}')
            
            # L.2. Generate Predictions and update the self.predictions attribute
            start = self.lookbacks[i] + timedelta(days=1)
            end = self.lookbacks[i + 1] if i < len(self.lookbacks) - 1 else None    # TODO: This seems to be causing problems with the number of predictions
            preds = pd.DataFrame(columns=assets)
            for name, frame in self.assets.items():
                try:
                    n_preds = self.generate_preds(frame.loc[start:end])
                    if (miss := self.update_freq - len(n_preds)) > 0 and i != 0 and i != len(self.lookbacks) - 1:   
                        # For datasets with assets that have different history lengths, we need to pad the front-end of the history with 0's
                        n_preds = np.pad(n_preds, (miss, 0), 'constant', constant_values=(0,0))
                    preds[name] = n_preds
                except ValueError:
                    preds[name] = 0
            
            try:
                self.predictions = pd.concat([self.predictions, preds], axis=0)
                print(f'Completed predictions for period {i + 1} of {len(self.lookbacks)}')
            except ValueError:
                print('ValueError raised, refer to below pandas objects for reference.')
                print(self.predictions, preds)
                return

        # After Loop:
        # Use the predictions and the returns data to calculate self.strat_rets attribute
        # TODO: Break this out into a separate method that calculates weights based on a selection
        try:
            start = self.lookbacks[0] + timedelta(days=1)
            self.predictions.index = self.dates_inter[self.dates_inter >= start]
            # Cut the function off here and move below to new strat returns method
            # counts = self.predictions.sum(axis=1).to_list()
            # counts = pd.Series([1/count if count != 0 else 0 for count in counts], index=self.predictions.index)
            # selections = self.predictions.multiply(counts, axis=0)
            # self.strat_rets = ((self.returns.loc[self.predictions.index] * selections)
            #                    .sum(axis=1)
            #                    .to_frame('ML-Strategy Returns'))
        except TypeError:
            print('A TypeError was raised in the final step. Test the final multiplication for strat returns.')

    @staticmethod
    def ridge_MV_optimization(returns, selections, sigma, ridge_penalty, look_back):
        """
        Computes a ridge-regularized mean-variance optimization for each row in the selections.
        The optimization is performed on the assets with positive returns in the row.
        The optimization is performed on the data from the previous look_back days.
        The optimization is performed with the following objective function:
        - Portfolio return - ridge_penalty * sum(weights^2)
        The optimization is subject to the following constraints:
        - Sum of weights = 1
        - Portfolio variance <= sigma
        The optimization is performed using the SLSQP method.
        The optimal weights are stored in a matrix with the same shape as the selections.
        The optimal weights are stored in the same order as the assets in the selections.
        
        :param returns: A DataFrame with the data for the assets. It takes shape (number of days, number of assets).
        :param selections: A DataFrame with the classifier selections for each asset. This must have the same shape as the data parameter.
        :param sigma: A float with the maximum portfolio variance.
        :param ridge_penalty: A float with the ridge penalty.
        :param look_back: An integer with the number of days to look back.
        """
        weight_matrix = [[0 for _ in range(selections.shape[1]-1)] for _ in range(selections.shape[0])]
        ind_dict = {f'Asset {i}':i-1 for i in range(1, len(selections.columns)+1)}
        diff = len(returns) - len(selections) - 1

        for i in range(0, selections.shape[0]):
            row = selections.iloc[i, 1:]
            positive_selections = row[row>0]
            # NOTE: positive_selections.index is a series, and the index is equal to the columns of the data DataFrame.
            # NOTE: target_matrix is a DataFrame with the data for the assets selected for the lookback period.
            target_matrix = returns.loc[diff+i-look_back+1:diff+i, positive_selections.index]
            
            # Using target_matrix to compute the covariance matrix and the mean return
            covariance_matrix = target_matrix.cov().values
            mean_return = target_matrix.mean().values
            num_of_assets = len(mean_return)
            if (num_of_assets==0):
                continue
            # optimization
            def objective(weights):
                portfolio_return = np.dot(weights, mean_return)
                ridge_term = ridge_penalty*np.sum(weights**2)
                # Why are we taking the negative of this term?
                return -(portfolio_return-ridge_term)
            def variance_constraint(weights):
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                return sigma-portfolio_variance
            constrains = [{'type':'eq', 'fun':lambda weights: np.sum(weights)-1}, 
                        {'type':'ineq', 'fun': variance_constraint}]
            bounds = tuple((0,1) for _ in range(num_of_assets))
            initial_weights = num_of_assets*[1/num_of_assets]
            
            result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constrains)
            optimal_weight = result.x if result.success else np.zeros(num_of_assets)
            
            for j in range(len(optimal_weight)):
                asset = positive_selections.index[j]
                ind = ind_dict[asset]
                weight_matrix[i][ind] = optimal_weight[j]

        return weight_matrix
    
    def calc_strat_rets(self, strat_name: str, selection: str, weighting: str, mvo_params: dict=None, momentum_period: int=None):
        """
        Records strategy returns based on the selections and weights selected
        :param strat_name: The name of the strategy
        :param selection: Choice of selections type.
            - 'momentum': Benchmark Strategy that selects assets that had positive returns over period
            - 'classifier': Strategy that selects assets based on the classifier predictions
            - None: Include all assets in the universe for the weighting step
        :param weighting: Choice of weighting methodology
            - 'equal': Equal weighting of assets
            - 'risk-parity': Scale weights based on inverse of asset volatility
            - 'ridge_MVO': Ridge-regularized mean-variance optimization
        :param mvo_params: Dictionary of parameters for the mean-variance optimization
            - 'sigma': Maximum portfolio variance
            - 'ridge_penalty': Ridge penalty
            - 'look_back': Number of days to look back
        """
        # Logic for obtaining selections dataframe
        if selection == 'classifier':
            selections = self.predictions
        elif selection == 'momentum':
            if momentum_period:
                lag = self.returns.rolling(window=momentum_period).apply(lambda x: np.prod(1 + x) - 1, raw=False)
                selections = pd.DataFrame(np.where(lag.shift(1) > 0, 1, 0), columns=self.returns.columns, index=self.returns.index)
            else:
                selections = pd.DataFrame(np.where(self.returns.shift(1) > 0, 1, 0), columns=self.returns.columns, index=self.returns.index)
        else:
            selections = pd.DataFrame(np.where(self.returns != 0, 1, 0), columns=self.returns.columns, index=self.returns.index)
        
        # Logic for calculating weights of selections
        counts = selections.sum(axis=1).to_list()
        if weighting == 'ridge_MVO':
            if (not mvo_params or
                'sigma' not in mvo_params or
                'ridge_penalty' not in mvo_params or
                'look_back' not in mvo_params):
                raise ValueError('Missing or mis-named parameters for the mean-variance optimization')
            rets = self.returns.reset_index()
            select = selections.reset_index()
            with parallel_backend('ray'):   # Attempting to speed up MVO with parallelism
                weights = self.ridge_MV_optimization(rets, select, **mvo_params)
            del rets, select
            weights = pd.DataFrame(weights, index=selections.index, columns=selections.columns).shift(1).dropna()
            self.mvo_weights = weights
        elif weighting == 'risk-parity':
            raise NotImplementedError('Risk Parity weighting has not been implemented yet. Please select another weighting method.')
        else:
            weights = pd.Series([1/count if count != 0 else 0 for count in counts], index=selections.index)
            del counts
            weights = selections.multiply(weights, axis=0)

        # Calculation of strategy returns using the weights and the returns data
        self.strat_rets[f'{strat_name} Strategy'] = ((self.returns.loc[selections.index] * weights)
                                                     .sum(axis=1))

    def performance_summary(self):
        # Generate a summary table and cumulative returns plot
        calc_cumulative_returns(self.strat_rets)
        return calc_summary_statistics(self.strat_rets, annual_factor=12, provided_excess_returns=True, 
                                correlations=False, keep_columns=['Annualized Mean', 'Annualized Vol',
                                                                    'Min','Max', 'Skewness', 'Excess Kurtosis',
                                                                    'Historical VaR', 'Historical CVaR',
                                                                    'Max Drawdown','Peak', 'Bottom',
                                                                    'Recover','Duration (days)']).T


class Classifier:
    random_state = 42
    def __init__(self, model, dim_red):
        self.pipeline = None
        self.model = model
        self.dim_red = dim_red
        self.model.random_state = Classifier.random_state
        self._params_hist = []
        
        # Attempting to speed up training with parallelism
        ray.init(address='auto', 
                 logging_level=logging.FATAL, 
                 log_to_driver=False, ignore_reinit_error=True)
        
    def train(self, features, target, param_grid=None):
        memory = Memory(location='cache_dir', verbose=0)
        pipeline = Pipeline([
            ('dim_red', self.dim_red),
            ('classifier', self.model)
        ], memory=memory)

        if param_grid:
            cv = StratifiedKFold(n_splits=5)
            grid_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=cv, scoring='f1',
                                            random_state=Classifier.random_state)
            with parallel_backend('ray'):   # Attempting to speed up training with parallelism
                grid_search.fit(features, target)
                # print(grid_search.best_params_)
            self._params_hist.append(grid_search.best_params_)
            self.pipeline = grid_search.best_estimator_
        else:
            with parallel_backend('ray'):
                self.pipeline = pipeline.fit(features, target)

    def predict(self, data):
        return self.pipeline.predict(data)


# Example Use (for debugging purposes):
if __name__ == '__main__':
    def assets_file_str_func(universe, assets):
        asset_nums = [asset.split(' ',1)[1] for asset in assets]
        return [universe + f'_{num}_' + 'features' for num in asset_nums]
    
    DATA_PATH = '/Users/austingalm/Documents/GitHub/fpi_project_lab_autumn2024/data/'
    CLASSIFIER_DATA_PATH = DATA_PATH + 'classifier_full/'
    ANALYSIS_PATH = '/Users/austingalm/Documents/GitHub/fpi_project_lab_autumn2024/analysis/'
    
    UNIVERSE = 'broad_asset'    # SELECT UNIVERSE HERE: 'equity_domestic', 'broad_asset', 'equity_global'
    if UNIVERSE == 'equity_domestic':
        ASSETS = ['Asset ' + str(i) for i in range(1, 47)]
        
        eqd_backtest = Backtest(model=RandomForestClassifier(), base_per='2004-04-10', update_freq=6)
        eqd_backtest.read_returns(data_path=DATA_PATH, universe_returns='equity_domestic_monthly_rets')
        eqd_backtest.compute_lookbacks(data_freq='monthly')
        # backtest.read_features(data_path=CLASSIFIER_DATA_PATH, features=FEATURES)
        eqd_backtest.read_data(data_path=DATA_PATH, assets=ASSETS, universe=UNIVERSE, file_str_func=assets_file_str_func)

        param_grid = {#'pca__n_components': [0.9],
                'classifier__n_estimators': [10], 
                'classifier__min_samples_split': [300], 
                'classifier__max_depth': [2],}    # 'classifier__class_weight':['balanced_subsample']
        eqd_backtest.record_strat_rets(assets=ASSETS, param_grid=param_grid)
    
    elif UNIVERSE == 'broad_asset':
        ASSETS = ['Asset ' + str(i) for i in range(1, 12)]

        backtest = Backtest(model=RandomForestClassifier(), base_per='2004-04-20', update_freq=26)
        backtest.read_returns(data_path=DATA_PATH, universe_returns='broad_assets_weekly_rets')
        backtest.compute_lookbacks()
        # backtest.read_features(data_path=CLASSIFIER_DATA_PATH, features=FEATURES)
        backtest.read_data(data_path=DATA_PATH, assets=ASSETS, universe=UNIVERSE, file_str_func=assets_file_str_func)

        param_grid = {#'pca__n_components': [0.9],
              'classifier__n_estimators': [10], 
              'classifier__min_samples_split': [300], 
              'classifier__max_depth': [2],}    # 'classifier__class_weight':['balanced_subsample']
        backtest.record_strat_rets(assets=ASSETS, param_grid=param_grid)

    else:
        ASSETS = ['Asset ' + str(i) for i in range(1, 12)]

        backtest = Backtest(model=RandomForestClassifier(), base_per='2007-04-20', update_freq=26)
        backtest.read_returns(data_path=DATA_PATH, universe_returns='equity_global_monthly_rets')
        backtest.compute_lookbacks()
        # backtest.read_features(data_path=CLASSIFIER_DATA_PATH, features=FEATURES)
        backtest.read_data(data_path=DATA_PATH, assets=ASSETS, universe=UNIVERSE, file_str_func=assets_file_str_func)

        param_grid = {#'pca__n_components': [0.9],
              'classifier__n_estimators': [10], 
              'classifier__min_samples_split': [300], 
              'classifier__max_depth': [2],}    # 'classifier__class_weight':['balanced_subsample']
        backtest.record_strat_rets(assets=ASSETS, param_grid=param_grid)


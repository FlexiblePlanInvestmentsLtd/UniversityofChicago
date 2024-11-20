import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

import sys
import os
from joblib import Memory
import joblib

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)
from analysis.utils import *

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
    def __init__(self, model, base_per, update_freq):
        self.model = Classifier(model)
        self.base_per = datetime.datetime.strptime(base_per, '%Y-%m-%d')
        self.assets = {}
        self.target = None
        self.update_freq = update_freq
        self.lookbacks = None
        self.dates_inter = None
        self.returns = None
        self.predictions = None
        self.strat_rets = None

    def read_returns(self, data_path, universe_returns):
        self.returns = pd.read_excel(data_path + universe_returns + '.xlsx', index_col=0, parse_dates=True)

    def read_data(self, data_path, universe, assets, file_str_func=None):
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
        target = target.loc[dates_inter]
        self.target = pd.DataFrame(np.where(target > 0, 1, 0), columns=target.columns, index=target.index)

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

    def record_strat_rets(self, assets, param_grid):    # TODO: Revise this after the feature pipeline is updated
        # Loop: 
        self.predictions = pd.DataFrame(columns=assets)
        for i in range(len(self.lookbacks)):
            # L.1. Train the model on data up until a lookback period
            features, target = self.build_training_dataset(self.lookbacks[i])
            self.train_model(features=features, target=target, 
                             param_grid=param_grid)#, dump_version=f'mod_backtest_{self.lookbacks[i]}')
            
            # L.2. Generate Predictions and update the self.predictions attribute
            preds = pd.DataFrame(columns=assets)
            for name, frame in self.assets.items():
                start = self.lookbacks[i] + timedelta(days=1)
                end = self.lookbacks[i + 1] if i < len(self.lookbacks) - 1 else None    # TODO: This seems to be causing problems with the number of predictions
                try:
                    preds[name] = self.generate_preds(frame.loc[start:end])
                except ValueError:
                    preds[name] = 0
            
            try:
                self.predictions = pd.concat([self.predictions, preds], axis=0)
            except ValueError:
                print('ValueError raised, refer to below pandas objects for reference.')
                print(self.predictions, preds)
                return

        # After Loop:
        # Use the predictions and the returns data to calculate self.strat_rets attribute
        try:
            start = self.lookbacks[0] + timedelta(days=1)
            self.predictions.index = self.dates_inter[self.dates_inter >= start]
            counts = self.predictions.sum(axis=1).to_list() # Equal weighting of assets -> update to Ridge MVO
            counts = pd.Series([1/count if count != 0 else 0 for count in counts], index=self.predictions.index)
            selections = self.predictions.multiply(counts, axis=0)
            self.strat_rets = ((self.returns.loc[self.predictions.index] * selections)
                               .sum(axis=1)
                               .to_frame('ML-Strategy Returns'))
        except TypeError:
            print('A TypeError was raised in the final step. Test the final multiplication for stat returns.')

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
    def __init__(self, model):
        self.pipeline = None
        self.model = model

    def train(self, param_grid, features, target):
        memory = Memory(location='cache_dir', verbose=0)
        pipeline = Pipeline([
            # ('scaler', MinMaxScaler()),
            # ('pca', PCA()),
            ('classifier', self.model)
        ], memory=memory)

        cv = StratifiedKFold(n_splits=5)
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv)
        grid_search.fit(features, target)
        # print(grid_search.best_params_)
        self.pipeline = grid_search.best_estimator_

    def predict(self, data):
        return self.pipeline.predict(data)

# Example Use:
if __name__ == '__main__':
    DATA_PATH = '/Users/austingalm/Documents/GitHub/fpi_project_lab_autumn2024/data/'
    ANALYSIS_PATH = '/Users/austingalm/Documents/GitHub/fpi_project_lab_autumn2024/analysis/'
    with open(ANALYSIS_PATH + '/classifier_features.txt', 'r') as file:
        FEATURES = [line.strip() for line in file]
    
    backtest = Backtest(model=RandomForestClassifier(), base_per='2020-04-20', update_freq=26)
    backtest.read_returns(data_path=DATA_PATH, universe_returns='broad_assets_weekly_rets')
    backtest.compute_lookbacks()
    backtest.read_features(data_path=DATA_PATH, features=FEATURES)
    
    assets = ['Asset 1', 'Asset 2', 'Asset 3', 'Asset 4', 'Asset 5', 'Asset 6', 
              'Asset 7', 'Asset 8', 'Asset 9', 'Asset 10', 'Asset 11']
    param_grid = {'n_estimators': 10, 'min_samples_split': 300, 'max_depth': 2, 'class_weight':'balanced_subsample'}
    backtest.record_strat_rets(assets=assets, param_grid=param_grid)
    # backtest.performance_summary()
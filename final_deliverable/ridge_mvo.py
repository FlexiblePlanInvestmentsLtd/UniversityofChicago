import pandas as pd
import numpy as np
from final_deliverable.utils_final import *

import os
pd.options.display.float_format = "{:,.4f}".format

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from scipy.optimize import minimize

class dynamic_ridge_optimizer:
    def __init__(self, data, cls_data, sigma, ridge_penalty, lookback_period, eval, annual_factor, ind_dict, start_date_lst=[], end_date_lst=[]):
        # eval is a parameter that user can choose from 'Sharpe' or 'Drawdown'
        # if 'Sharpe', means we want to max(Sharpe) to determine optimal lookback
        # if 'Drawdown', means we want to min(Drawdown) to determine optimal lookback
        self.data = data
        self.classification_data = cls_data
        self.sigma = sigma
        self.ridge_penalty = ridge_penalty
        self.lookback_period = lookback_period
        self.start_date_lst = start_date_lst
        self.end_date_lst = end_date_lst
        self.start_index = []
        self.end_index = []
        self.eval = eval
        self.annual_factor = annual_factor
        self.ind_dict = ind_dict
        self.weight_matrix = [[0 for _ in range(self.data.shape[1]-1)] for _ in range(self.data.shape[0])]
        self.optimal_lookback_lst = []

    def calculate_cum_return(self, ret):
        res = []
        cum_sum = 1
        for i in range(len(ret)):
            cum_sum*=(1+ret[i])
            res.append(cum_sum)
        return res
    
    def find_yearly_date(self):
        date_lst = self.data['Date']
        i = 0
        while (i<=self.data.shape[0]-2):
            self.start_date_lst.append(date_lst[i])
            self.start_index.append(i)
            j = i+1
            while (j<=self.data.shape[0]-1):
                if (date_lst[j]-date_lst[i]).days>=365:
                    self.end_date_lst.append(date_lst[j])
                    self.end_index.append(j)
                    break
                else:
                    j+=1
            i = j
        self.end_date_lst.append(date_lst[self.data.shape[0]-2])
        self.end_index.append(self.data.shape[0]-2)

    def single_dynamic_ridge_optimization(self, start_ind, look_back):
        # for the weeks before the look_back period, we just do equal weight for the assets
        backtest_weight_matrix = [[0 for _ in range(self.data.shape[1]-1)] for _ in range(start_ind+1)]
        for i in range(look_back-1):
            row = self.classification_data.iloc[i, 1:].values
            count = 0
            for j in range(self.classification_data.shape[1]-1):
                if (row[j]>0): # consider it
                    count+=1
                    backtest_weight_matrix[i][j] = 1
            for j in range(self.classification_data.shape[1]-1):
                if backtest_weight_matrix[i][j]==1:
                    backtest_weight_matrix[i][j] = 1/count

        for i in range(look_back-1, start_ind+1):
            row = self.classification_data.iloc[i, 1:]
            positive_returns = row[row>0]
            target_matrix = self.data.loc[i-look_back+1:i, positive_returns.index]
        
            covariance_matrix = target_matrix.cov().values
            mean_return = target_matrix.mean().values
            num_of_assets = len(mean_return)
            if (num_of_assets==0):
                continue
            # optimization
            def objective(weights):
                portfolio_return = np.dot(weights, mean_return)
                ridge_term = self.ridge_penalty*np.sum(weights**2)
                return -(portfolio_return-ridge_term)
            def variance_constraint(weights):
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                return self.sigma-portfolio_variance
            constrains = [{'type':'eq', 'fun':lambda weights: np.sum(weights)-1}, 
                        {'type':'ineq', 'fun': variance_constraint}]
            bounds = tuple((0,1) for _ in range(num_of_assets))
            initial_weights = num_of_assets*[1/num_of_assets]

            result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constrains)
            optimal_weight = result.x if result.success else np.zeros(num_of_assets)
            
            for j in range(len(optimal_weight)):
                asset = positive_returns.index[j]
                ind = self.ind_dict[asset]
                backtest_weight_matrix[i][ind] = optimal_weight[j]

        weight_DataFrame = pd.DataFrame(backtest_weight_matrix, index=self.data['Date'][0:start_ind+1], columns=[f'Asset {i}' for i in range(1,self.data.shape[1])])
        weight_DataFrame = weight_DataFrame.iloc[0:-1, :]
        weights = weight_DataFrame.values
        data1 = self.data.iloc[1:start_ind+1, :]
        data1 = data1.set_index('Date')
        week_returns = data1.values
        portfolio_returns = np.sum(weights*week_returns, axis=1)
        final_stats = summary_statistics_annualized(portfolio_returns, self.annual_factor)
        if (self.eval=='Sharpe'):
            return final_stats['Sharpe']
        else:
            return final_stats['Max Drawdown']

    def find_optimum_lookback(self, start_ind):
        if self.eval=='Sharpe': # choose Sharpe as the evaluation metric
            sharpe_lst = np.array([])
            for look_back in self.lookback_period:
                sharpe = self.single_dynamic_ridge_optimization(start_ind, look_back)
                sharpe_lst = np.append(sharpe_lst, sharpe)
            self.optimal_lookback_lst.append(self.lookback_period[np.argmax(sharpe_lst)])
            return self.lookback_period[np.argmax(sharpe_lst)] # the optimal lookback in the backtest
        else: # choose Drawdown as the evaluation metric
            drawdown_lst = np.array([])
            for look_back in self.lookback_period:
                drawdown = self.single_dynamic_ridge_optimization(start_ind, look_back)
                drawdown_lst = np.append(drawdown_lst, drawdown)
            self.optimal_lookback_lst.append(self.lookback_period[np.argmin(drawdown_lst)])
            return self.lookback_period[np.argmin(drawdown_lst)]

    def optimize_weight_in_next_year(self, start_ind, end_ind):
        optimal_lookback = self.find_optimum_lookback(start_ind)
        for i in range(start_ind, end_ind+1):
            row = self.classification_data.iloc[i, 1:]
            positive_returns = row[row>0]
            target_matrix = self.data.loc[i-optimal_lookback+1:i, positive_returns.index]
        
            covariance_matrix = target_matrix.cov().values
            mean_return = target_matrix.mean().values
            num_of_assets = len(mean_return)
            if (num_of_assets==0):
                continue
            # optimization
            def objective(weights):
                portfolio_return = np.dot(weights, mean_return)
                ridge_term = self.ridge_penalty*np.sum(weights**2)
                return -(portfolio_return-ridge_term)
            def variance_constraint(weights):
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                return self.sigma-portfolio_variance
            constrains = [{'type':'eq', 'fun':lambda weights: np.sum(weights)-1}, 
                        {'type':'ineq', 'fun': variance_constraint}]
            bounds = tuple((0,1) for _ in range(num_of_assets))
            initial_weights = num_of_assets*[1/num_of_assets]

            result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constrains)
            optimal_weight = result.x if result.success else np.zeros(num_of_assets)

            for j in range(len(optimal_weight)):
                asset = positive_returns.index[j]
                ind = self.ind_dict[asset]
                self.weight_matrix[i][ind] = optimal_weight[j]

    def loop_to_update_optimized_weight(self):
        self.find_yearly_date()
        for i in range(1, len(self.start_index)):
            start_ind = self.start_index[i]
            end_ind = self.end_index[i]
            self.optimize_weight_in_next_year(start_ind, end_ind)

    def report_summary_statistics(self, classification_type):
        self.loop_to_update_optimized_weight() # we get the optimized weight matrix
        weight_DataFrame = pd.DataFrame(self.weight_matrix, index=self.data['Date'], columns=[f'Asset {i}' for i in range(1, self.data.shape[1])])
        weight_DataFrame = weight_DataFrame.iloc[self.start_index[1]:-1, :] 
        save_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        file_name = f'{classification_type}_{self.eval}_weight.xlsx'
        save_path = os.path.join(save_dir, file_name)
        weight_DataFrame.to_excel(save_path)
        weights = weight_DataFrame.values
        data1 = self.data.iloc[self.start_index[1]+1:, :]
        data1 = data1.set_index('Date')
        week_returns = data1.values
        portfolio_returns = np.sum(weights*week_returns, axis=1)
        x = np.array(data1.index)
        y = self.calculate_cum_return(portfolio_returns)

        num_of_assets = self.data.shape[1]-1
        benchmark_portfolio = np.sum((1/num_of_assets)*week_returns, axis=1)
        y1 = self.calculate_cum_return(benchmark_portfolio)
        # Another portfolio, equally weighted those assets that we predict a positive return
        weight_df = self.classification_data.set_index('Date').iloc[self.start_index[1]:, :]
        num_assets = weight_df.sum(axis=1)
        new_weight = weight_df.div(num_assets, axis=0).values
        classification_equal = np.sum(new_weight*week_returns, axis=1)
        y2 = self.calculate_cum_return(classification_equal)

        fig = plt.figure(figsize=(8, 6))
        plt.plot(x, y, label='Ridge MVO')
        plt.plot(x, y1, label='benchmark')
        plt.plot(x, y2, label='cls equal')
        plt.xlabel('Date')
        plt.ylabel('Cumulative return')
        plt.legend()
        final_stats = summary_statistics_annualized(portfolio_returns, self.annual_factor)
        cls_stats = summary_statistics_annualized(classification_equal, self.annual_factor)
        benchmark_stats = summary_statistics_annualized(benchmark_portfolio, self.annual_factor)
        return pd.concat([final_stats, cls_stats, benchmark_stats], keys=['cls_ridge', 'cls_equal', 'benchmark'], axis=0)

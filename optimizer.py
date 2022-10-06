"""
Portfolio Optimizer
Michael Pham, Summer 2022
"""

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import getFamaFrenchFactors as gff
import csv
import matplotlib.pyplot as plt
import random
import random as rd


def read_in_csv():
  """
  reads in spreadsheet csv of tickers
  returns list of tickers
  """
  csv_df = pd.read_csv ('stocks.csv')
  stocks = []
  for i in range(len(csv_df)):
    stocks.append(csv_df.loc[i, 'Tickers'])


  return stocks



def get_data_set(investable_universe_ls, yf_format_start_date, yf_format_end_date):
  """
  takes list of tickers, start/end dates as parameters
  returns data frame of historical price data for all tickers
  """
  yfinance_input_string = ""
  for ticker in investable_universe_ls:
    yfinance_input_string += (ticker + " ")
  df = yf.download(yfinance_input_string, start=yf_format_start_date, end=yf_format_end_date, auto_adjust = True).Close
  

  return df



def get_dates(simulation_date, years_data):
  """
  from simulation start data and years of historical data requested,
  returns the start and end dates for the historical data
  """
  end_year = simulation_date[:4]
  end_month = simulation_date[5:]

  start_year = str(int(end_year) - years_data)
  start_month = end_month

  yf_format_start_date = start_year + '-' + start_month + '-' + '01'
  yf_format_end_date = end_year + '-' + end_month + '-' + '01'


  return yf_format_start_date, yf_format_end_date



def fama_french(stock_data_resample, ticker, start, end, factor):
  """
  calculates expected return for each ticker
  """

  #fetch fama french spreads from fama french library
  df_ff5_monthly = gff.famaFrench5Factor(frequency='m')

  #merges fama french spreads with historical price data and formats into one data frame
  df_ff5_monthly = df_ff5_monthly.loc[(df_ff5_monthly['date_ff_factors'] >= start) & (df_ff5_monthly['date_ff_factors'] < end)]
  df = pd.merge(stock_data_resample, df_ff5_monthly, left_on='Date', right_on='date_ff_factors')
  df['Excess'] = df[ticker] - df['RF']
  df['MKT'] = df['Mkt-RF']

  #runs when running fama french five factor model
  if factor == 'five':
    #runs multiple linear regression to find beta coefficients
    fama_lm = smf.ols(formula = 'Excess ~ MKT + SMB + HML + RMW + CMA', data = df).fit()
    intercept, b1, b2, b3, b4, b5 = fama_lm.params
    rf = df['RF'].mean()

    #gets average fama french spreads for period
    market_premium = df['Mkt-RF'].mean()
    size_premium = df['SMB'].mean()
    value_premium = df['HML'].mean()
    quality_premium = df['RMW'].mean()
    investment_premium = df['CMA'].mean()

    #uses beta coefficients and fama french spreads to get expected monthly return
    expected_monthly_return = (rf + b1 * market_premium + b2 * size_premium + b3 * value_premium + b4 * quality_premium + b5 * investment_premium)
  

    return expected_monthly_return * 12



  #runs when running fama french three factor model
  elif factor == 'three':
    #runs multiple linear regression to find beta coefficients
    fama_lm = smf.ols(formula = 'Excess ~ MKT + SMB + HML', data = df).fit()
    intercept, b1, b2, b3 = fama_lm.params
    rf = df['RF'].mean()
    
    #gets average fama french spreads for period
    market_premium = df['Mkt-RF'].mean()
    size_premium = df['SMB'].mean()
    value_premium = df['HML'].mean()

    #uses beta coefficients and fama french spreads to get expected monthly return
    expected_monthly_return = (rf + b1 * market_premium + b2 * size_premium + b3 * value_premium)
    
  
    return expected_monthly_return * 12

  #runs when running historical mean model
  elif factor == 'mean':
    return df[ticker].mean() * 12



def calculate_covariance(investable_universe_ls, stock_data_resample):
  """
  calculates and returns covariance matrix
  """
  covariance_matrix = stock_data_resample.cov()


  return covariance_matrix



def calculate_bmatrix(investable_universe_ls, df, reference_return):
  """
  calculates and returns b matrix
  """
  
  #gets t, an integer that represents time periods
  t = len(df)

  #calculate t x 1 vector (greek letter i) that fits needs of equation
  filler_vector = []
  for h in range(t):
    filler_vector.append([1])
  filler_vector = np.array(filler_vector)

  #calculate t x n Matrix R where there are n variables
  r_matrix = df
  
  #calculate 1 x t vector e
  e = []
  for h in range(len(df.columns)):
    e.append(reference_return)
  e_vector = np.array([e,])
  e_matrix = np.matmul(filler_vector, e_vector)

  #calculate t x n Matrix B
  b_matrix = ( 1/np.sqrt(t) ) * (r_matrix - e_matrix)
  

  return b_matrix



def calculate_portfolio_return(weights, mu):
  """
  retuns expected portfolio return
  """
  transpose_weights = np.transpose(weights)
  

  return transpose_weights.dot(mu)



def calculate_portfolio_variance(weights, covariance_matrix):
  """
  returns portfolio variance
  """

  #multiplies transposed weights vector by covariance matrix by weights vector to get portfolio variance
  portfolio_variance = np.transpose(weights)@covariance_matrix@weights

  #takes square root of variance to get standard deviation
  portfolio_volatility = np.sqrt(portfolio_variance)


  return portfolio_volatility



def calculate_portfolio_semivariance(weights, b_matrix):
  """
  returns portfolio semivariance
  """
  weights_vector = np.array(weights)

  #create matrix A which is (x^t B^t)^-
  a = np.transpose(weights_vector)@np.transpose(b_matrix) #need transpose
  num = a._get_numeric_data()
  num[num > 0] = 0

  #create matrix B which is (Bx)^-
  b = b_matrix@weights_vector
  num = b._get_numeric_data()
  num[num > 0] = 0

  a = a.dropna()
  b = b.dropna()
  
  #multiple matrix A and matrix B to get semivariance
  semivariance = a@b


  return np.sqrt(semivariance)



def monthly_resample(data):
  stock_data_adj_close = data.dropna()
  #stock_data_adj_close = data.fillna(method='ffill', inplace = True)


  #RIGHT HERE

  #print(stock_data_adj_close)
  # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
  #   print(stock_data_adj_close)

  resample_monthly_df = stock_data_adj_close.resample('M').agg(lambda x: x[-1]/x[1] - 1)
  
  #print(resample_monthly_df)
  
  resample_daily_df = stock_data_adj_close.pct_change()
  


  return resample_monthly_df, resample_daily_df

def get_target(portfolios_df, target):
  while True:
    try:
      #print(portfolios_df)

      rf = 0.025

      portfolios_over_target_df = portfolios_df.loc[(portfolios_df['return'] >= target)]
      portfolios_over_target_df['sharpe'] = (portfolios_over_target_df['return']-rf)/portfolios_over_target_df['volatility']
      portfolios_over_target_df = portfolios_over_target_df.sort_values(by=['sharpe'], ascending=False)
      target_portfolio = portfolios_over_target_df.iloc[0]

      #print(target_portfolio)
      return target_portfolio.loc['weight'], target_portfolio.loc['return'], target_portfolio.loc['volatility']
      break
    except:
      target -= 0.005
      pass


def get_sharpe(portfolios_df):
  rf = 0.025
  sharpe_portfolio = portfolios_df.iloc[((portfolios_df['return']-rf)/portfolios_df['volatility']).idxmax()]
  print(sharpe_portfolio)
  return sharpe_portfolio.loc['weight'], sharpe_portfolio.loc['return'], sharpe_portfolio.loc['volatility']

def results_to_csv(weights, tickers, file_name):
  with open(file_name, 'w') as f:
      for i in range(len(tickers)):
          f.write("%s,%s\n"%(tickers[i],weights[i]))


def main(simulation_date, years_data, min_position_size, max_position_size, mu_method, optimization_method, target, variance):

  #get list of stocks from csv
  investable_universe_ls = read_in_csv()

  #get data from start_date-years_data to start_date, returns df of all data
  yf_format_start_date, yf_format_end_date = get_dates(simulation_date, years_data)
  
  
  data_set = get_data_set(investable_universe_ls, yf_format_start_date, yf_format_end_date)

  #resample monthly stock data
  for ticker in investable_universe_ls:
    drop_na = data_set[ticker.upper()].dropna()
    drop_last = data_set[ticker.upper()]
    drop_last.drop(drop_last.tail(1).index,inplace=True)
    if drop_last.equals(drop_na) == False:
      print('ERROR WITH TICKER:')
      print(ticker)


  resample_monthly_df, resample_daily_df = monthly_resample(data_set)
  
  #get expected return for all stocks, returns df
  exp_return = []
  for i in range(len(investable_universe_ls)):
    
    mu_individual = fama_french(resample_monthly_df, investable_universe_ls[i], yf_format_start_date, yf_format_end_date, mu_method)
    
    exp_return.append(mu_individual)
  mu = pd.Series(exp_return, index=investable_universe_ls)

  mu_df = mu.to_list()
  results_to_csv(mu_df, investable_universe_ls, 'expectedreturns.csv')

  

  #get covariance
  # variance = 'semivariance'
  if variance == 'semivariance':
    bmatrix = calculate_bmatrix(investable_universe_ls, resample_daily_df, 0)
  elif variance == 'covariance':
    covariance_matrix = calculate_covariance(investable_universe_ls, resample_daily_df)
    
    print(covariance_matrix)

  
  #SIM - turn into function later:

  number_stocks = len(investable_universe_ls)

  all_returns = [] #stores expected returns
  all_volatilities = [] #stores volatilities
  all_weights = [] #store weights

  #set amount of portfolios to simulate
  portfolios = 50000



  min_basis_points = min_position_size * 10000
  max_basis_points = max_position_size * 10000
  increment = 10

  for portfolio in range(portfolios):
    # weights = []
    # for element in range(number_stocks):
    #   value = np.random.choice(np.arange(0, 11), p=prob_weighting)
    #   weights.append(value)
    # weights /= np.sum(weights)
    weights = []
    weight_total = 0
    for i in range(number_stocks):
      value = rd.randrange(min_basis_points, max_basis_points + increment, increment)

      # options = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
      # probability = [0.25, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.26]
      # value = rd.choices(options, probability)

      value /= 10000
      weights.append(value)

      weight_total += value
      
      if i == number_stocks - 2:
        weights.append(1 - weight_total)
        break


      if weight_total >= 1 - max_basis_points/10000:
        weights.append(1 - weight_total)

        remaining = number_stocks - i - 2
        for i in range(remaining):
          weights.append(0)
        break
    rd.shuffle(weights)

    
    if (all(x <= max_position_size for x in weights)):
      all_weights.append(weights)

      portfolio_expected_return = calculate_portfolio_return(weights, mu)
      all_returns.append(portfolio_expected_return)

      
      if variance == 'semivariance':
        portfolio_volatility = calculate_portfolio_semivariance(weights, bmatrix)
      elif variance == 'covariance':
        portfolio_volatility = calculate_portfolio_variance(weights, covariance_matrix)

      all_volatilities.append(portfolio_volatility)

  # print(weights)
  # print(portfolio_expected_return)
  # print(portfolio_volatility)

  # print(all_returns)
  # print('VOL')
  # print(all_volatilities)
  
  #print('WEIGHT')
  #print(all_weights)

  portfolios_dict = {
    'return': all_returns,
    'volatility': all_volatilities,
    'weight': all_weights
  }
  
  portfolios_df = pd.DataFrame(portfolios_dict)

  print(portfolios_df)
  
  
  #GRAPH
  # portfolios_df.plot.scatter(x = 'volatility', y = 'return', marker = 'o', color = 'y', s = 15, alpha = 0.5, grid = True, figsize = [8,8])
  # plt.show()

  if optimization_method == 'sharpe':
    sharpe_weights, best_return, best_volatility = get_sharpe(portfolios_df)
    results_to_csv(sharpe_weights, investable_universe_ls, 'results.csv')
  # print(sharpe_weights)
  # results_to_csv(sharpe_weights, investable_universe_ls, 'sharpe.csv')
  
  elif optimization_method == 'target':
    target_weights, best_return, best_volatility = get_target(portfolios_df, target)
    results_to_csv(target_weights, investable_universe_ls, 'results.csv')
  # print(target_weights)
  # results_to_csv(target_weights, investable_universe_ls, 'target.csv')



  # dict_results = {}
  # for i in range(len(investable_universe_ls)):
  #   dict_results[investable_universe_ls[i]] = sharpe_weights[i]
  # #print(dict_results)

  # dict_results_target = {}
  # for i in range(len(investable_universe_ls)):
  #   dict_results[investable_universe_ls[i]] = target_weights[i]
  # #print(target_weights)

  return best_return, best_volatility
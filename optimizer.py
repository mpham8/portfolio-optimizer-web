#from statistics import covariance
import yfinance as yf
import pandas as pd
import numpy as np
# from pypfopt import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
import statsmodels.formula.api as smf
import getFamaFrenchFactors as gff
import csv
import matplotlib.pyplot as plt
import random
import random as rd


def read_in_csv():
  
  csv_df = pd.read_csv ('stocks.csv')
  stocks = []
  for i in range(len(csv_df)): #len(df)
    stocks.append(csv_df.loc[i, 'Tickers'])
  

  return stocks


def get_data_set(investable_universe_ls, yf_format_start_date, yf_format_end_date):

  yfinance_input_string = ""
  for ticker in investable_universe_ls:
    yfinance_input_string += (ticker + " ")
  
  df = yf.download(yfinance_input_string, start=yf_format_start_date, end=yf_format_end_date, auto_adjust = True).Close
  

  return df



def get_dates(simulation_date, years_data):
  #START DATE = 'YYYY-MM'

  end_year = simulation_date[:4]
  end_month = simulation_date[5:]


  start_year = str(int(end_year) - years_data)
  start_month = end_month

  #data_start_date = start_month + '-' + end

  yf_format_start_date = start_year + '-' + start_month + '-' + '01'
  yf_format_end_date = end_year + '-' + end_month + '-' + '01'


  return yf_format_start_date, yf_format_end_date



def fama_french(stock_data_resample, ticker, start, end, factor):
  
  df_ff5_monthly = gff.famaFrench5Factor(frequency='m')
  df_ff5_monthly = df_ff5_monthly.loc[(df_ff5_monthly['date_ff_factors'] >= start) & (df_ff5_monthly['date_ff_factors'] < end)]
  
  #print(stock_data_resample) #WRONG HERE
  #print(df_ff5_monthly)
  
  df = pd.merge(stock_data_resample, df_ff5_monthly, left_on='Date', right_on='date_ff_factors')
  df['Excess'] = df[ticker] - df['RF']
  df['MKT'] = df['Mkt-RF']
  #print(df)

  if factor == 'five':
    fama_lm = smf.ols(formula = 'Excess ~ MKT + SMB + HML + RMW + CMA', data = df).fit()
    intercept, b1, b2, b3, b4, b5 = fama_lm.params
    rf = df['RF'].mean()

    market_premium = df['Mkt-RF'].mean()
    size_premium = df['SMB'].mean()
    value_premium = df['HML'].mean()
    quality_premium = df['RMW'].mean()
    investment_premium = df['CMA'].mean()
    expected_monthly_return = (rf + b1 * market_premium + b2 * size_premium + b3 * value_premium + b4 * quality_premium + b5 * investment_premium)
  
    return expected_monthly_return * 12

  elif factor == 'three':
    fama_lm = smf.ols(formula = 'Excess ~ MKT + SMB + HML', data = df).fit()
    intercept, b1, b2, b3 = fama_lm.params
    rf = df['RF'].mean()
    #print(rf)

    # print(b1)
    # print(b2)
    # print(b3)


    market_premium = df['Mkt-RF'].mean()
    size_premium = df['SMB'].mean()
    value_premium = df['HML'].mean()
    expected_monthly_return = (rf + b1 * market_premium + b2 * size_premium + b3 * value_premium)
    

    return expected_monthly_return * 12

  elif factor == 'mean':
    return df[ticker].mean() * 12



def calculate_covariance(investable_universe_ls, stock_data_resample):
  #standard_deviation_df = stock_data_resample.std()
  print(stock_data_resample)
  
  covariance_matrix = stock_data_resample.cov()

  return covariance_matrix

def calculate_bmatrix(investable_universe_ls, df, reference_return):
  t = len(df)
  # print(t)

  filler_vector = []
  for h in range(t):
    filler_vector.append([1])
  filler_vector = np.array(filler_vector)
  # print(filler_vector)


  r_matrix = df
  # print(r_matrix)

  e = []

  for h in range(len(df.columns)):
    e.append(reference_return)
  # print(e)


  e_vector = np.array([e,])
  # print(e_vector)

  e_matrix = np.matmul(filler_vector, e_vector)
  # print(e_matrix)

  b_matrix = ( 1/np.sqrt(t) ) * (r_matrix - e_matrix)
  # print(b_matrix)


  #print(b_matrix)
  return b_matrix


def calculate_portfolio_return(weights, mu):
  transpose_weights = np.transpose(weights)
  # print(weights)
  # print(mu)
  return transpose_weights.dot(mu)

def calculate_portfolio_variance(weights, covariance_matrix):
  portfolio_variance = np.transpose(weights)@covariance_matrix@weights
  portfolio_volatility = np.sqrt(portfolio_variance)

  return portfolio_volatility

def calculate_portfolio_semivariance(weights, b_matrix):
  
  weights_vector = np.array(weights)

  a = np.transpose(weights_vector)@np.transpose(b_matrix) #need transpose
  num = a._get_numeric_data()
  num[num > 0] = 0

  b = b_matrix@weights_vector
  num = b._get_numeric_data()
  num[num > 0] = 0

  a = a.dropna()
  b = b.dropna()
  # print(a)
  # print(b)

  semivariance = a@b

  # print(semivariance)

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
  #print(data_set)

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
    #ticker = investable_universe_ls[i]
    mu_individual = fama_french(resample_monthly_df, investable_universe_ls[i], yf_format_start_date, yf_format_end_date, mu_method)
    #print(mu_individual)
    exp_return.append(mu_individual)
  mu = pd.Series(exp_return, index=investable_universe_ls)

  mu_df = mu.to_list()
  results_to_csv(mu_df, investable_universe_ls, 'expectedreturns.csv')

  # file_name = 'expectedreturns-' + str(simulation_date) + '.csv'
  # f = open(file_name, "w")
  # f.close()
  # results_to_csv(mu_df, investable_universe_ls, file_name)

  #print(mu)
  # for x in mu:
  #   print(x)
  print(mu.mean())

  #get covariance
  # variance = 'semivariance'
  if variance == 'semivariance':
    bmatrix = calculate_bmatrix(investable_universe_ls, resample_daily_df, 0)
  elif variance == 'covariance':
    covariance_matrix = calculate_covariance(investable_universe_ls, resample_daily_df)
  
  #covariance_matrix = calculate_semicovariance(investable_universe_ls, resample_daily_df)
  
    print(covariance_matrix)

  
  #SIM - turn into function later:

  #weights = [0.25, 0.35, 0.4]

  number_stocks = len(investable_universe_ls)

  all_returns = [] #stores expected returns
  all_volatilities = [] #stores volatilities
  all_weights = [] #store weights

  portfolios = 50000


  #max_position_size

  # for i in range(portfolios):
  #   # weights = np.random.random(number_stocks)
    
  #   weights = []
  #   num_zero = random.randint(0, number_stocks//2)
  #   num_nonzero = number_stocks - num_zero

  #   for i in range(num_zero):
  #     weights.append(0)
  #   for i in range(num_nonzero):
  #     value = random.randint(1, 10)
  #     weights.append(value)

  # for i in range(portfolios):
  #   weights = []
  #   accumulator = 0
  #   max_int = int(max_position_size * 10) 
  #   for j in range(number_stocks):
  #     if (1 - accumulator) < max_position_size:
  #       weights.append(1 - accumulator)
  #       break
  #     else:
  #       value = random.randint(0,max_int)/10
  #       weights.append(value)
  #       accumulator += value
  #   if len(weights) != 0:
  #     zeros = 1 - len(weights)
  #     for k in range(zeros):
  #       weights.append(0)
  #   if np.sum(weights) == 1:
  #     #weights = weights/np.sum(weights)
  #     random.shuffle(weights)
  
  prob_weighting = [0.275, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.275]
  #prob_weighting = [0.3, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]
  #prob_weighting = [0.75, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.16]

  #prob_weighting = [0.32, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.32]

  #UNCOMMENT

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


    # weights = np.random.random(number_stocks)
    # weights /= np.sum(weights)
    #print(weights)
  
  
  # for portfolio in range(portfolios):
  #   weights = []
  #   accumulator = 0
  #   for element in range(number_stocks):
  #     if 1 - accumulator >= max_position_size:
  #       odds_max = random.choice([0,1,2,3])
  #       if odds_max == 0:
  #         value = max_position_size
  #       else:
  #         value = random.uniform(0, max_position_size)
  #       weights.append(value)
  #       accumulator += value
  #     else:
  #       value = 1 - accumulator
  #       weights.append(value)
  #       zeros_to_fill = number_stocks - len(weights)
  #       for i in range(zeros_to_fill):
  #         weights.append(0)
  #       random.shuffle(weights)
  #       break
    
    
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

  #target_weights = get_target(portfolios_df, target)
  #print(target_weights)

  # rf = 0.0244

  # sharpe = 

  #maximise sharpe ratio/target


  #write results to csv


#main('2016-06', 5, 0.005, 0.02, 'three', 'target', 0.15)
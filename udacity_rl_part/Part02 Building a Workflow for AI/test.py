import pandas as pd
import func_lib 

import os

path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(path, "historical_prices.csv")
print(csv_path)

if __name__ == '__main__':
  #  Create a variable named 'historical_prices' to store historical price data between the dates'2005-01-01' and '2015-01-01'. Use the function 'create_hist_prices()' from 'func_lib' to generate or retrieve the historical prices.
  # historical_prices = func_lib.create_hist_prices(start_date = '2005-01-01', end_date = '2015-01-01')
  historical_prices = func_lib.load_historical_prices(csv_path, start_date = '2005-01-01', end_date = '2015-01-01')

  # Create a list named 'list_of_momentums' and add 1 to the list
  list_of_momentums = [1]
  # print(historical_prices)
  # Compute the total returns using the historical prices and the list of momentums. Use the 'compute_returns()' function from 'func_lib'. Pass 'historical_prices' and 'list_of_momentums' as arguments to 'func_lib.compute_returns()'.
  # - Store the result in a variable named 'total_returns'.
  total_returns     = func_lib.compute_returns(historical_prices, list_of_momentums)
  # Use the 'dropna()' method on the 'total_returns' DataFrame.
  total_returns.dropna(inplace=True) 
  
  print(total_returns.head())
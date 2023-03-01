from datetime import datetime
from typing import Callable, List

import numpy as np

from common import Stock, StockRecord


def calculate_annualized_standard_deviation(
    stock_records: List[StockRecord],
) -> float:
    """
    S_i is the stock price at time i
    R_i is the stock price return at time i
    n is the number of returns
    sd is the standard deviation of return
    sd_annual is annualized standard deviation of return
    t is daily data representing 252 business days per year
    R_i = (S_i / S_i-1) - 1
    avg_i = nΣi=1 R_i / n
    acc_diff = nΣi=1 (R_i - avg_R)
    sd = sqrt((1/n-1) * acc_diff)
    sd_annual = sd * sqrt(t)
    """
    if len(stock_records) <= 1:
        raise Exception(f'Stock records size too small {len(stock_records)}')
    
    prices = np.array([record.price for record in stock_records])
    returns = prices[1:] / prices[0:-1] - 1
    avg_return = np.average(returns)
    n = len(stock_records) - 1
    sd = np.sqrt(np.sum(np.power(returns - avg_return, 2) / (n-1)))
    sd_annual = sd * np.sqrt(252)
    return np.round(sd_annual, 5)


def calculate_annualized_covariance(*args, **kwargs) -> float:
    pass


def calculate_correlation_coefficient(*args, **kwargs) -> float:
    pass


def run(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
):
    for stock in stocks:
        stock_records = read_history_records(stock.code)
        sd_annual = calculate_annualized_standard_deviation(stock_records)
        print(f'Stock {stock.code:<5} {sd_annual}')
    calculate_annualized_covariance()
    calculate_annualized_covariance()

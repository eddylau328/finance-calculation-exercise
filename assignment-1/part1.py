from typing import Callable, List

import numpy as np

from common import Stock, StockRecord

def calculate_stock_price_returns(stock_records: List[StockRecord]) -> np.ndarray:
    if len(stock_records) <= 1:
        raise Exception(f'Stock records size too small {len(stock_records)}')
    
    prices = np.array([record.price for record in stock_records])
    return prices[1:] / prices[0:-1] - 1


def calculate_annualized_standard_deviation(
    stock_records: List[StockRecord],
) -> float:
    """
    S_i       : the stock price at time i
    R_i       : the stock price return at time i
    n         : the number of returns
    sd        : the standard deviation of return
    sd_annual : annualized standard deviation of return
    t         : daily data representing 252 business days per year

    R_i = (S_i / S_i-1) - 1
    avg_i = nΣi=1 R_i / n
    acc_diff = nΣi=1 (R_i - avg_R)
    sd = sqrt((1/n-1) * acc_diff)
    sd_annual = sd * sqrt(t)

    """
    returns = calculate_stock_price_returns(stock_records)
    sd = np.std(returns)
    sd_annual = sd * np.sqrt(252)
    return np.round(sd_annual, 5)


def calculate_annualized_covariance(
    a_stock_records: List[StockRecord],
    b_stock_records: List[StockRecord],
) -> float:
    """
    a_returns    : the stock price returns for Stock A
    a_avg_return : the average stock price return for Stock A
    b_returns    : the stock price returns for Stock B
    b_avg_return : the average stock price return for Stock B
    n            : the number of returns

    annualized_covariance = nΣi=1 [(R_a_i - avg_R_a)(R_b_i - avg_R_b)] * 252 / (n - 1)
    """
    a_returns = calculate_stock_price_returns(a_stock_records)
    b_returns = calculate_stock_price_returns(b_stock_records)
    covariance = np.cov(a_returns, b_returns)[0][1]
    return covariance * 252


def calculate_correlation_coefficient(
    a_stock_records: List[StockRecord],
    b_stock_records: List[StockRecord],
) -> float:
    """
    sd_annual_a           : the stock annualized standard deviation for Stock A
    sd_annual_b           : the stock annualized standard deviation for Stock B
    annualized_covariance : the annualized covariance between Stock A and Stock B

    correlation_coefficient = sd_ab / (sd_a * sd_b)
    """
    sd_annual_a = calculate_annualized_standard_deviation(a_stock_records)
    sd_annual_b = calculate_annualized_standard_deviation(b_stock_records)
    annualized_covariance = calculate_annualized_covariance(
        a_stock_records,
        b_stock_records,
    )
    return annualized_covariance / (sd_annual_a * sd_annual_b)


def run(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
):
    print('Annualized Standard Deviation')
    for stock in stocks:
        stock_records = read_history_records(stock.code)
        sd_annual = calculate_annualized_standard_deviation(stock_records)
        print(f'Stock {stock.code:<5} {sd_annual}')

    print()

    print('Annualized Covariance')
    for a_stock in stocks:
        for b_stock in stocks:
            if a_stock.code == b_stock.code:
                continue
            a_stock_records = read_history_records(a_stock.code)
            b_stock_records = read_history_records(b_stock.code)
            annualized_covariance = calculate_annualized_covariance(
                a_stock_records,
                b_stock_records,
            )
            print(f'Stock {a_stock.code:>4} vs {b_stock.code:<5} {annualized_covariance}')

    print()

    print('Correlation Coefficient')
    for a_stock in stocks:
        for b_stock in stocks:
            if a_stock.code == b_stock.code:
                continue
            a_stock_records = read_history_records(a_stock.code)
            b_stock_records = read_history_records(b_stock.code)
            annualized_covariance = calculate_correlation_coefficient(
                a_stock_records,
                b_stock_records,
            )
            print(f'Stock {a_stock.code:>4} vs {b_stock.code:<5} {annualized_covariance}')

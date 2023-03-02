from typing import Callable, List

import numpy as np

from common import Stock, StockRecord
from part1 import calculate_stock_price_returns


def calculate_expected_return(stock_records: List[StockRecord]) -> float:
    returns = calculate_stock_price_returns(stock_records)
    return np.mean(returns)


def calculate_standard_deviation(stock_records: List[StockRecord]) -> float:
    returns = calculate_stock_price_returns(stock_records)
    return np.std(returns)


def calculate_portfolio_expected_return(
    weights: List[float],
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
):
    expected_returns = np.array([
        calculate_expected_return(read_history_records(stock.code))
        for stock in stocks
    ])
    return np.dot(expected_returns, np.array(weights))


def calculate_portfolio_standard_deviation(
    weights: List[float],
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
):
    standard_deviations = np.array([
        calculate_standard_deviation(read_history_records(stock.code))
        for stock in stocks
    ])
    return np.dot(standard_deviations, np.array(weights))


def calculate_global_minimum_variance_portfolio(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]], 
) -> List[float]:
    """
    Use Lagrange multipliers to solve the problem

    sd_p^2 = nΣi=1(w_i^2 * sd_i^2) + nΣi=1(nΣj=1(w_i * w_j * sd_i_j))
    
    2 constraints:
    avg_r_p = nΣi=1(w_i * avg_r_i)
    sum_of_weights = nΣi=1(w_i) = 1

    Introduce Lagrange multipliers, with λ_1, λ_2

    L = 1/2 * sd_p^2 - λ_1 * avg_r_p - λ_2 * sum_of_weights

    Differentiating L w.r.t. to each w_i:

    ∂L/∂w_i = 0, i = 1 ... N

    Resulting to N + 2 equations:

    w_i * sd_i^2 + nΣj=1(sd_ij * w_j) - λ_1 * avg_r_i - λ_2 = 0, i = 1 ... N
    avg_r_p = nΣi=1(w_i * avg_r_i)
    nΣi=1(w_i) = 1

    Solve the linear matrix equation
    A = [
            [sd_i^2 * w_i]
        ]

    Resulting with parameters:
    λ_1, λ_2, w_i where i = 1 ... N
    """
    expected_returns = np.array([
        calculate_expected_return(read_history_records(stock.code))
        for stock in stocks
    ])
    standard_deviations = np.array([
        calculate_standard_deviation(read_history_records(stock.code))
        for stock in stocks
    ])


def run(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
):
    weights = [0.1] * len(stocks)

    portfolio_expected_return = calculate_portfolio_expected_return(
        weights,
        stocks,
        read_history_records,
    )
    portfolio_standard_deviation = calculate_portfolio_standard_deviation(
        weights,
        stocks,
        read_history_records,
    )
    print(f'Expected Return of EW    {portfolio_expected_return}')
    print(f'Standard Deviation of EW {portfolio_standard_deviation}')

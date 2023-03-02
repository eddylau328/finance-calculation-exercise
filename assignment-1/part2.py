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

from typing import List, Callable, Tuple
from string import Template

import numpy as np

from common import (
    Stock,
    StockRecord,
)


def calculate_stock_price_log_returns(
    stock_prices: List[float],
) -> np.ndarray:
    if len(stock_prices) <= 1:
        raise Exception(f'Stock records size too small {len(stock_prices)}')
    prices = np.array(stock_prices)
    return np.log(prices[1:] / prices[0:-1])


def calculate_realized_volatility(stock_prices: List[float]):
    log_returns = calculate_stock_price_log_returns(stock_prices)
    # numpy std, ddof=1 -> n - 1
    return np.std(log_returns, ddof=1) * np.sqrt(252)


def calculate_correlation_coefficient(
    stock_1_prices: List[StockRecord],
    stock_2_prices: List[StockRecord],
) -> float:
    stock_1_log_returns = calculate_stock_price_log_returns(stock_1_prices)
    stock_2_log_returns = calculate_stock_price_log_returns(stock_2_prices)
    stock_1_diff = stock_1_log_returns - np.mean(stock_1_log_returns)
    stock_2_diff = stock_2_log_returns - np.mean(stock_2_log_returns)
    n = len(stock_1_prices)
    std_1_1 = calculate_realized_volatility(stock_1_prices) 
    std_2_2 = calculate_realized_volatility(stock_2_prices) 
    covar_1_2 = np.sum(np.multiply(stock_1_diff, stock_2_diff)) * 252 / (n - 1)
    return covar_1_2 / (std_1_1 * std_2_2)


def part_1_i(
    stock_1_records: List[StockRecord],
    stock_2_records: List[StockRecord],
) -> Tuple[float, float]:
    stock_1_volatility = calculate_realized_volatility([r.price for r in stock_1_records])
    stock_2_volatitliy = calculate_realized_volatility([r.price for r in stock_2_records])
    return stock_1_volatility, stock_2_volatitliy


def print_part_1_i(results: Tuple[float, float]):
    template = Template('Stock S$index realized volatility = $result')
    print('1i) Realized Volatilities')
    for i, result in enumerate(results):
        print(template.safe_substitute(index=i+1, result=result))
    print()


def part_1_ii(
    stock_1_records: List[StockRecord],
    stock_2_records: List[StockRecord],
) -> float:
    return calculate_correlation_coefficient(
        [r.price for r in stock_1_records],
        [r.price for r in stock_2_records],
    )


def print_part_1_ii(result: float):
    print('1ii) Correlation Coefficient')
    print(f'correlation coef. of S1 and S2 = {result}')
    print()


def run(
    stock_1: Stock,
    stock_2: Stock,
    read_stock_data: Callable[[int], List[StockRecord]],
):
    stock_1_records = read_stock_data(stock_1.code)
    stock_2_records = read_stock_data(stock_2.code)

    volatilities = part_1_i(stock_1_records, stock_2_records)
    print_part_1_i(volatilities)

    corr_coef = part_1_ii(stock_1_records, stock_2_records)
    print_part_1_ii(corr_coef)

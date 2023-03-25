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


def part_1_2_i(
    stock_1_records: List[StockRecord],
    stock_2_records: List[StockRecord],
    volatilities:Tuple[float, float], 
    corr_coef: float
):
    S10 = stock_1_records[-1].price
    S20 = stock_2_records[-1].price
    T = 1
    dt = 1/180
    r = 3.75/100

    N = int(T / dt)    
    n_paths = [1000, 10000, 100000, 500000]

    for n in n_paths:
        S1 = np.zeros((n, N+1))
        S2 = np.zeros((n, N+1))
        S1[:, 0] = S10
        S2[:, 0] = S20 

        for i in range(1, N+1):
            z1 = np.random.normal(0 ,1, size=n)
            z2 = corr_coef*z1 + np.sqrt(1 - corr_coef**2) * np.random.normal(0 ,1, size=n)

            S1[:, i] = S1[:, i-1] * (1 + r * dt + volatilities[0] * z1 * np.sqrt(dt))
            S2[:, i] = S2[:, i-1] * (1 + r * dt + volatilities[1] * z2 * np.sqrt(dt))

        S11 = S1[:, int(N/2)]
        S12 = S1[:,-1]
        S21 = S2[:, int(N/2)]
        S22 = S2[:,-1]

        B1 = np.maximum(S11 / S10, S21 / S20)
        B2 = np.maximum(S12 / S10, S22 / S20)
        A = (B1 + B2) / 2
        payoff = np.maximum(A - 1, 0)
        option_price = np.mean(payoff) 
        print(n, 'path', option_price)


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

    part_1_2_i(stock_1_records, stock_2_records, volatilities, corr_coef)

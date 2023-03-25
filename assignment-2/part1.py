from typing import List, Callable, Tuple
from string import Template
import time

import numpy as np
import matplotlib.pyplot as plt

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
    stock_1_prices: List[float],
    stock_2_prices: List[float],
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


def calculate_covariance_matrix(
    stock_1_prices: List[float],
    stock_2_prices: List[float],
):
    stock_1_log_returns = calculate_stock_price_log_returns(stock_1_prices)
    stock_2_log_returns = calculate_stock_price_log_returns(stock_2_prices)
    stock_1_diff = stock_1_log_returns - np.mean(stock_1_log_returns)
    stock_2_diff = stock_2_log_returns - np.mean(stock_2_log_returns)
    n = len(stock_1_prices)
    std_1_1 = calculate_realized_volatility(stock_1_prices) 
    std_2_2 = calculate_realized_volatility(stock_2_prices) 
    covar_1_2 = np.sum(np.multiply(stock_1_diff, stock_2_diff)) * 252 / (n - 1)
    cov_matrix = np.full(shape=(2, 2), fill_value=0.0)
    cov_matrix[0][0] = np.power(std_1_1, 2)
    cov_matrix[0][1] = covar_1_2
    cov_matrix[1][1] = np.power(std_2_2, 2)
    cov_matrix[1][0] = covar_1_2
    return cov_matrix


def calculate_mean_returns(
    stock_1_prices: List[float],
    stock_2_prices: List[float],
):
    stock_1_log_returns = calculate_stock_price_log_returns(stock_1_prices)
    stock_2_log_returns = calculate_stock_price_log_returns(stock_2_prices)
    return np.transpose(np.mean(np.vstack((stock_1_log_returns, stock_2_log_returns)), axis=1))


def geo_paths(S, T, r, sigma, steps, N):
    """
    Inputs
    # S = Current stock Price
    # T = Time to maturity 1 year = 1, 1 months = 1/12
    # r = risk free interest rate
    # sigma = volatility 
    
    Output
    # [steps,N] Matrix of asset paths 
    """
    dt = T/steps
    ST = np.log(S) +  np.cumsum(((r - sigma**2/2)*dt +\
                              sigma*np.sqrt(dt) * \
                              np.random.normal(size=(steps,N))),axis=0)
    
    return np.exp(ST)


def run_monte_carlo_discretization_scheme(
    stock_1_start_price: float,
    stock_2_start_price: float,
    stock_1_volatility: float,
    stock_2_volatility: float,
    corr: float,
    maturity: float,
    risk_free_rate: float,
    total_time_frame: int,
    total_simulation: int,
) -> float:
    overall_option_price = 0
    dt = maturity / total_time_frame
    for _ in range(total_simulation):
        X = np.random.normal(size=(total_time_frame, 2))
        E_1 = X[:, 0]
        E_2 = corr * X[:, 0] + X[:, 1] * np.sqrt(1 - corr**2)
        S_1 = stock_1_start_price * np.exp((risk_free_rate - stock_1_volatility**2 / 2) * dt + stock_1_volatility * E_1 * np.sqrt(dt))
        S_2 = stock_2_start_price * np.exp((risk_free_rate - stock_2_volatility**2 / 2) * dt + stock_2_volatility * E_2 * np.sqrt(dt))
        S_1_0, S_1_1, S_1_2 = stock_1_start_price, S_1[0], S_1[1]
        S_2_0, S_2_1, S_2_2 = stock_2_start_price, S_2[0], S_2[1]
        B_1 = np.max(np.array([S_1_1 / S_1_0, S_2_1 / S_2_0]))
        B_2 = np.max(np.array([S_1_2 / S_1_0, S_2_2 / S_2_0]))
        A = (B_1 + B_2) / 2
        overall_option_price += np.max(np.array([A - 1, 0]))
    return overall_option_price / total_simulation


# PART 1
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


# PART 2
def part_2_i(
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


def part_2_ii(
    stock_1_records: List[StockRecord],
    stock_2_records: List[StockRecord],
) -> List[Tuple[int, float]]:
    stock_1_prices = [r.price for r in stock_1_records]
    stock_2_prices = [r.price for r in stock_2_records]
    stock_1_volatility = calculate_realized_volatility([r.price for r in stock_1_records])
    stock_2_volatility = calculate_realized_volatility([r.price for r in stock_2_records])
    corr = calculate_correlation_coefficient(stock_1_prices, stock_1_prices)
    maturity = 1
    risk_free_rate = 0.0375
    total_time_frame = 2
    num_of_paths = [1000, 10000, 100000, 500000]
    results = []
    for path in num_of_paths:
        # start_time = time.time()
        # print('path', path)
        average_option_price = run_monte_carlo_discretization_scheme(
            stock_1_prices[-1],
            stock_2_prices[-1],
            stock_1_volatility,
            stock_2_volatility,
            corr,
            maturity,
            risk_free_rate,
            total_time_frame,
            path,
        )
        results.append((path, average_option_price))
        # print("--- %s seconds ---" % (np.round(time.time() - start_time, 5)))

    return results


def print_part_2_ii(
    results: List[Tuple[int, float]],
):
    print('2ii) Monte Carlo discretization scheme with time steps N = 2')
    template = Template('$num_of_path paths: option price = $option_price')
    for num_of_path, option_price in results:
        print(template.safe_substitute(
            num_of_path=f'{num_of_path:<6}',
            option_price=f'{option_price}',
        ))


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

    part_2_i(stock_1_records, stock_2_records, volatilities, corr_coef)
    # print_part_2_i(results)

    results = part_2_ii(stock_1_records, stock_2_records)
    print_part_2_ii(results)

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


def calculate_covariance_matrix(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
) -> np.ndarray:
    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])
    return np.cov(returns_matrix)


def calculate_portfolio_expected_return(
    weights: List[float],
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
) -> float:
    expected_returns = np.array([
        calculate_expected_return(read_history_records(stock.code))
        for stock in stocks
    ])
    return np.dot(expected_returns, np.array(weights))


def calculate_portfolio_standard_deviation(
    weights: List[float],
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
) -> float:
    """
    result: nΣi((w_i)**2(q_i)**2)+nΣj(nΣk(w_j * w_k * q_jk))

    use the matrix method sd_p^2 = (w^T)(cov)(w)
    """
    weights = np.array(weights)
    covariance_matrix = calculate_covariance_matrix(
        stocks,
        read_history_records,
    )
    portfolio_variance = np.dot(weights, covariance_matrix)
    portfolio_variance = np.dot(portfolio_variance, weights)
    return np.sqrt(portfolio_variance)


def calculate_global_minimum_variance_portfolio(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]], 
) -> List[float]:
    """
    Use Lagrange multipliers to solve the problem

    sd_p^2 = nΣi=1(w_i^2 * sd_i^2) + nΣi=1(nΣj=1(w_i * w_j * cov_i_j))
    
    2 constraints:
    avg_r_p = nΣi=1(w_i * avg_r_i)
    sum_of_weights = nΣi=1(w_i) = 1

    Introduce Lagrange multipliers, with λ_1, λ_2

    L = 1/2 * sd_p^2 - λ_1 * (nΣi=1(w_i * avg_r_i) - avg_r_p) - λ_2 * (nΣi=1(w_i) - 1)

    Differentiating L w.r.t. to each w_i:

    ∂L/∂w_i = 0, i = 1 ... N

    Resulting to N + 2 equations:

    w_i * sd_i^2 + nΣj=1(cov_i_j * w_j) - λ_1 * avg_r_i - λ_2 = 0, i = 1 ... N
    avg_r_p = nΣi=1(w_i * avg_r_i)
    nΣi=1(w_i) = 1

    Solve the linear equation
    [
        (sd_1^2 + cov_1_1) * w_1 + cov_1_2 * w_2 + ... + cov_1_N * w_N - avg_r_1 * λ_1 - λ_2 = 0
        ...
        cov_N_1 * w_1 + ... + (sd_N^2 + cov_N_N) * w_N - avg_r_N * λ_1 - λ_2 = 0
        avg_r_1 * w_1 + ... + avg_r_N * w_N = avg_r_p
        w_1 + ... + w_N = 1
    ]

        AX = Y
    A^-1AX = A^-1Y
         X = A^-1Y

    w_1 = A^-1_1_11 * avg_r_p + A^-1_1_12
    w_2 = A^-1_2_11 * avg_r_p + A^-1_2_12
    ...
    w_N = A^-1_N_11 * avg_r_p + A^-1_N_12
    λ_1 = A^-1_N+1_11 * avg_r_p + A^-1_N+1_12
    λ_2 = A^-1_N+2_11 * avg_r_p + A^-1_N+2_12

    Resulting with parameters:
    λ_1, λ_2, w_i where i = 1 ... N
    """
    returns = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])
    expected_returns = np.array([
        calculate_expected_return(read_history_records(stock.code))
        for stock in stocks
    ])
    standard_deviations = np.array([
        calculate_standard_deviation(read_history_records(stock.code))
        for stock in stocks
    ])
    covariance_matrix = np.cov(returns)
    linear_matrix = None
    # First N linear equations
    for i in range(len(stocks)):
        # initialize
        equation = np.ones(len(stocks)+2)
        # w_1 to w_N
        for j in range(len(stocks)):
            if i == j:
                equation[j] *= standard_deviations[j] + covariance_matrix[i][j]
            else:
                equation[j] *= covariance_matrix[i][j]
        # λ_1, λ_2 setup
        equation[len(stocks)] *= -expected_returns[i]
        equation[len(stocks)+1] *= -1

        if i == 0:
            linear_matrix = equation
        else:
            linear_matrix = np.vstack([linear_matrix, equation])

    # Last 2 linear equations
    linear_matrix = np.vstack([
        linear_matrix,
        np.concatenate((expected_returns, np.zeros(2, np.float64)), axis=None),
    ])
    linear_matrix = np.vstack([
        linear_matrix,
        np.concatenate((np.ones(len(stocks), np.float64), np.zeros(2, np.float64)), axis=None),
    ])
    linear_matrix_inv = np.linalg.inv(linear_matrix)

    results = None
    for i in range(len(stocks)):
        # position 11, 12 = array index 10, 11
        equation = np.array([linear_matrix_inv[i][10], linear_matrix_inv[i][11]])
        if i == 0:
            results = equation
        else:
            results = np.vstack([results, equation])

    for i in range(len(stocks)):
        respresentation = f'Stock {stocks[i].code:>4}   w_{i} = '
        respresentation += f'{np.round(results[i][0], 4)} * avg_r_p '
        if np.sign(results[i][1]) == 1:
            respresentation += '+ '
        else:
            respresentation += '- '
        respresentation += f'{np.abs(np.round(results[i][1], 4))}'
        print(respresentation)


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

    calculate_global_minimum_variance_portfolio(stocks, read_history_records)
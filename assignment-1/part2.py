from typing import Callable, List
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt

from common import Stock, StockRecord, GlobalMinimumVariancePortfolio
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
) -> GlobalMinimumVariancePortfolio:
    """
    Reference: (not all correct)
    https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf

    Use Lagrange multipliers to solve the problem

    sd_p^2 = nΣi=1(w_i^2 * sd_i^2) + nΣi=1(nΣj=1(w_i * w_j * cov_i_j))
    
    1 constraints:
    sum_of_weights = nΣi=1(w_i) = 1

    Introduce Lagrange multipliers, with λ

    L = sd_p^2 - λ * (nΣi=1(w_i) - 1)

    Differentiating L w.r.t. to each w_i:

    ∂L/∂w_i = 0, i = 1 ... N
      ∂L/∂λ = 0

    Resulting to N + 1 equations:

    [ 2Σ 1 ] [ w ]  = [ 0 ]
    [ 1  0 ] [ λ ]    [ 1 ]

    [ w ] = [ 2Σ 1 ]^-1 [ 0 ]
    [ λ ]   [ 1  0 ]    [ 1 ]

    w = [w_1 w_2 ... w_n]
    Σ = covariance matrix

    """
    covariance_matrix = calculate_covariance_matrix(stocks, read_history_records)
    linear_matrix = np.hstack((2 * covariance_matrix, np.ones((len(stocks), 1), np.float64)))
    linear_matrix = np.vstack((linear_matrix, np.ones(len(stocks)+1, np.float64)))
    linear_matrix[len(stocks)][len(stocks)] = 0
    linear_matrix_inv = np.linalg.inv(linear_matrix)
    linear_matrix_ans = np.append(np.zeros(len(stocks), np.float64), 1.0)
    minimum_weights = np.dot(linear_matrix_inv, linear_matrix_ans)[0:10]
    minimum_expected_return = calculate_portfolio_expected_return(
        list(minimum_weights),
        stocks,
        read_history_records,
    )
    minimum_variance = reduce(np.dot, [minimum_weights, covariance_matrix, minimum_weights])
    return GlobalMinimumVariancePortfolio(
        standard_deviation=np.sqrt(minimum_variance),
        variance=minimum_variance,
        expected_return=minimum_expected_return,
        weights=list(minimum_weights),
    )

    results = None
    for i in range(len(stocks)):
        # position N+1, N+2 = array index N, N+1
        equation = np.array([linear_matrix_inv[i][len(stocks)], linear_matrix_inv[i][len(stocks)+1]])
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
    
    # sd_p^2 = nΣi=1(w_i^2 * sd_i^2) + nΣj=1(nΣk=1(w_j * w_k * cov_j_k))
    # sd_p^2 = c1 + c2 * r_p + c3 * r_p^2
    # index 0 is constant
    # index 1 is r_p
    # index 2 is r_p^2
    equation = np.zeros(3, np.float64)

    # solve for constant
    for i in range(len(stocks)):
        equation[0] += results[i][1]**2 * standard_deviations[i]**2
        for j in range(len(stocks)):
            for k in range(len(stocks)):
                equation[0] += results[j][1] * results[k][1] * covariance_matrix[j][k]

    # solve for r_p
    for i in range(len(stocks)):
        equation[1] += 2 * results[i][0] * results[i][1] * standard_deviations[i]**2
        for j in range(len(stocks)):
            for k in range(len(stocks)):
                equation[1] += results[j][0] * results[k][1] * covariance_matrix[j][k]
                equation[1] += results[k][0] * results[j][1] * covariance_matrix[j][k]

    # solve for r_p^2
    for i in range(len(stocks)):
        equation[2] += results[i][0]**2 * standard_deviations[i]**2
        for j in range(len(stocks)):
            for k in range(len(stocks)):
                equation[2] += results[j][0] * results[k][0] * covariance_matrix[j][k]

    print()
    respresentation = ''
    for i, variable in enumerate(['', 'r_p', 'r_p^2']):
        if np.sign(equation[i]) == -1:
            respresentation += ' - '
        elif i != 0:
            respresentation += ' + '
        respresentation += f'{np.abs(np.round(equation[i], 4))}'
        if variable:
            respresentation += f' * {variable}'

    respresentation = f'sd_p = sqrt({respresentation})'
    print(respresentation)

    print()
    print('Minimum expected return: ', np.round(np.min(expected_returns),4))
    print('Maximum expected return: ', np.round(np.max(expected_returns),4))

    # generate random weights
    test_data_sizes = 10000
    test_weights = np.random.rand(test_data_sizes, len(stocks))
    test_weights = test_weights / test_weights.sum(axis=1)[:, np.newaxis]
    # sd_p^2 = nΣi=1(w_i^2 * sd_i^2) + nΣi=1(nΣj=1(w_i * w_j * cov_i_j))
    test_sd_p = np.zeros(test_data_sizes, np.float64)
    for k in range(test_data_sizes):
        for i in range(len(stocks)):
            for j in range(len(stocks)):
                test_sd_p[k] += test_weights[k][i] * test_weights[k][j] * covariance_matrix[i][j]
    test_sd_p += np.dot(test_weights**2, standard_deviations**2)
    test_sd_p = np.sqrt(test_sd_p)
    #  avg_r_p = nΣi=1(w_i * avg_r_i)
    test_r_p = np.dot(test_weights, expected_returns)

    r_p = np.linspace(
        np.min(expected_returns),
        np.max(expected_returns),
        1000,
    )
    sd_p = np.sqrt(equation[0] + equation[1] * r_p + equation[2] * r_p**2)

    plt.plot(sd_p, r_p)
    plt.scatter(test_sd_p, test_r_p)
    fk = 0
    for i in range(len(stocks)):
        for j in range(len(stocks)):
            fk += global_minimum_variance_weights[i] * global_minimum_variance_weights[j] * covariance_matrix[i][j]
    fk += np.dot(global_minimum_variance_weights**2, standard_deviations**2)
    fk = np.sqrt(fk)
    plt.scatter(
        # [np.dot(np.dot(global_minimum_variance_weights, covariance_matrix), global_minimum_variance_weights)],
        [fk],
        [np.dot(global_minimum_variance_weights, expected_returns)],
        color='red',
    )
    plt.xlabel("Portfolio Standard Deviation")
    plt.ylabel("Portfolio Return")
    plt.show()


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

    print()

    global_mv_portfolio = calculate_global_minimum_variance_portfolio(stocks, read_history_records)
    print('Global minimum variance portfolio')
    print(f'Portfolio variance           {global_mv_portfolio.variance}')
    print(f'Portfolio standard deviation {global_mv_portfolio.standard_deviation}')
    print(f'Portfolio expected return    {global_mv_portfolio.expected_return}')
    print('Portfolio weights:')
    for i, stock in enumerate(stocks):
        print(f'Stock {stock.code:>5} weight = {global_mv_portfolio.weights[i]}')

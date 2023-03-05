from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt

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
        equation = np.zeros(len(stocks)+2)
        # w_1 to w_N
        for j in range(len(stocks)):
            if i == j:
                equation[j] = standard_deviations[j]**2 + covariance_matrix[i][j]
            else:
                equation[j] = covariance_matrix[i][j]
        # λ_1, λ_2 setup
        equation[len(stocks)] = -expected_returns[i]
        equation[len(stocks)+1] = -1

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

    # update fuck
    test = np.hstack((2 * covariance_matrix, np.ones((len(stocks), 1), np.float64)))
    test = np.vstack((test, np.ones(len(stocks)+1, np.float64)))
    test[len(stocks)][len(stocks)] = 0
    test_inv = np.linalg.inv(test)
    test_ans = np.append(np.zeros(len(stocks), np.float64), 1.0)
    global_minimum_variance_weights = np.dot(test_inv, test_ans)[0:10]
    print(global_minimum_variance_weights)
    print(np.sum(global_minimum_variance_weights))

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
    calculate_global_minimum_variance_portfolio(stocks, read_history_records)

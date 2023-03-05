from typing import Callable, List, Optional
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt

from common import Stock, StockRecord, GlobalMinimumVariancePortfolio
from part1 import (
    calculate_annualized_standard_deviation,
    calculate_annualized_covariance,
)


def calculate_covariance_matrix(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
) -> np.ndarray:
    covariance_matrix = np.zeros((len(stocks), len(stocks)), np.float64)
    for i, stocks_a in enumerate(stocks):
        for j, stocks_b in enumerate(stocks):
            covariance_matrix[i][j] = calculate_annualized_covariance(
                read_history_records(stocks_a.code),
                read_history_records(stocks_b.code),
            )
    return covariance_matrix


def calculate_portfolio_expected_return(
    weights: List[float],
    stocks: List[Stock],
) -> float:
    expected_returns = np.array([
        stock.expected_return for stock in stocks
    ])
    return np.dot(expected_returns, np.array(weights))


def calculate_portfolio_standard_deviation(
    weights: List[float],
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
    covariance_matrix: Optional[np.ndarray] = None,
) -> float:
    """
    result: nΣi((w_i)**2(q_i)**2)+nΣj(nΣk(w_j * w_k * q_jk))

    use the matrix method sd_p^2 = (w^T)(cov)(w)
    """
    weights = np.array(weights)
    if type(covariance_matrix) is not np.ndarray:
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
    covariance_matrix: Optional[np.ndarray] = None,
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
    if type(covariance_matrix) is not np.ndarray:
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
    )
    minimum_variance = reduce(np.dot, [minimum_weights, covariance_matrix, minimum_weights])
    return GlobalMinimumVariancePortfolio(
        standard_deviation=np.sqrt(minimum_variance),
        variance=minimum_variance,
        expected_return=minimum_expected_return,
        weights=list(minimum_weights),
    )

def calculate_efficient_frontier(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
    covariance_matrix: Optional[np.ndarray] = None,
):
    """
    Reference: (not all correct)
    https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf

    Use Lagrange multipliers to solve the problem

    sd_p^2 = nΣi=1(w_i^2 * sd_i^2) + nΣi=1(nΣj=1(w_i * w_j * cov_i_j))
    
    1 constraints:
    r_p = nΣi=1(w_i * r_i)

    Introduce Lagrange multipliers, with λ

    L = sd_p^2 + λ_1 * (nΣi=1(w_i * r_i) - r_p) + λ_2 * (nΣi=1(w_i) - 1)

    Differentiating L w.r.t. to each w_i:

    ∂L/∂w_i = 0, i = 1 ... N
      ∂L/∂λ = 0

    Resulting to N + 1 equations:

    [ 2Σ r  1 ] [  w  ]    [  0  ]
    [ r  0  0 ] [ λ_1 ]  = [ r_p ]
    [ 1  0  0 ] [ λ_2 ]    [  1  ]

    [  w  ]   [ 2Σ r  1 ]^-1 [  0  ]
    [ λ_1 ] = [ r  0  0 ]    [ r_p ]
    [ λ_2 ]   [ 1  0  0 ]    [  1  ]

    w = [w_1 w_2 ... w_n]
    Σ = covariance matrix
    r = expected returns

    """
    N = len(stocks)
    if type(covariance_matrix) is not np.ndarray:
        covariance_matrix = calculate_covariance_matrix(stocks, read_history_records)

    expected_returns = np.array([stock.expected_return for stock in stocks])
    linear_matrix = np.hstack((2 * covariance_matrix, np.transpose(np.array([expected_returns])), np.ones((N, 1), np.float64)))
    linear_matrix = np.vstack((linear_matrix, np.concatenate((expected_returns, np.zeros(2, np.float64)))))
    linear_matrix = np.vstack((linear_matrix, np.concatenate((np.ones(N, np.float64), np.zeros(2, np.float64)))))
    linear_matrix_inv = np.linalg.inv(linear_matrix)

    # weights in linear equation format, w = m * r_p + c
    # weights store [m, c]
    weight_linear_equations = np.zeros((N, 2), np.float64)
    for i in range(N):
        # index N = pos N + 1
        # index N + 1 = pos N + 2
        weight_linear_equations[i][0] = linear_matrix_inv[i][N]
        weight_linear_equations[i][1] = linear_matrix_inv[i][N+1]
        
    """
    sd_p^2 = nΣi=1(w_i^2 * sd_i^2) + nΣi=1(nΣj=1(w_i * w_j * cov_i_j))

    sd_p^2 = 2 * w_1^2 * cov_1_1 + ... + 2 * w_N^2 * cov_N_N
             + 2 * w_1 * w_2 * cov_1_2 + ... + 2 * w_N * w_N-1 * cov_N_N-1

    first part:
    2 * w_1^2 * cov_1_1
    = 2 * cov_1_1 * (m_1 * r_p + c_1)^2
    = 2 * cov_1_1 * (m_1^2 * r_p^2 + 2 * m_1 * c_1 * r_p + c_1^2)
    = 2 * cov_1_1 * m_1^2 * r_p^2
      + 4 * cov_1_1 * m_1 * c_1 * r_p
      + 2 * cov_1_1 * c_1^2

    second part:
    2 * w_1 * w_2 * cov_1_2
    = 2 * cov_1_2 * (m_1 * r_p + c_1) * (m_2 * r_p + c_2)
    = 2 * cov_1_2 * [m_1 * m_2 * r_p^2 + (m_1 * c_2 + m_2 * c_1) * r_p + c_1 * c_2]
    = 2 * cov_1_2 * m_1 * m_2 * r_p^2
      + 2 * cov_1_2 * (m_1 * c_2 + m_2 * c_1) * r_p
      + 2 * cov_1_2 * c_1 * c_2

    Combine:
    sd_p^2 = A * r_p^2 + B * r_p + C
    """
    A, B, C = 0, 0, 0
    for i in range(N):
        # first part
        A += 2 * covariance_matrix[i][i] * weight_linear_equations[i][0]**2
        B += 4 * covariance_matrix[i][i] * weight_linear_equations[i][0] * weight_linear_equations[i][1]
        C += 2 * covariance_matrix[i][i] * weight_linear_equations[i][1]**2
        # second part
        for j in range(N):
            if i == j:
                continue
            sub_part_a = 1
            sub_part_a *= 2 * covariance_matrix[i][j]
            sub_part_a *= weight_linear_equations[i][0]
            sub_part_a *= weight_linear_equations[j][0]
            A += sub_part_a
            sub_part_b = 1
            sub_part_b *= 2 * covariance_matrix[i][j]
            sub_part_b *= (
                weight_linear_equations[i][0] * weight_linear_equations[j][1] +
                weight_linear_equations[j][0] * weight_linear_equations[i][1]
            )
            B += sub_part_b
            sub_part_c = 1
            sub_part_c *= 2 * covariance_matrix[i][j]
            sub_part_c *= weight_linear_equations[i][1]
            sub_part_c *= weight_linear_equations[j][1]
            C += sub_part_c

    print(f'sd_p^2 = {A} * r_p^2 + {B} * r_p + {C}')
    r_p = np.linspace(
        np.min(expected_returns),
        np.max(expected_returns),
        1000,
    )
    sd_p = np.sqrt(1/2*(A * r_p**2 + B * r_p + C))

    portfolio = calculate_global_minimum_variance_portfolio(stocks, read_history_records)

    test_data_sizes = 100000
    test_weights = np.random.rand(test_data_sizes, len(stocks))
    test_weights = test_weights / test_weights.sum(axis=1)[:, np.newaxis]

    # sd_p^2 = nΣi=1(w_i^2 * sd_i^2) + nΣi=1(nΣj=1(w_i * w_j * cov_i_j))
    test_sd_p = np.zeros(test_data_sizes, np.float64)
    for k in range(test_data_sizes):
        for i in range(len(stocks)):
            for j in range(len(stocks)):
                test_sd_p[k] += test_weights[k][i] * test_weights[k][j] * covariance_matrix[i][j]

    test_sd_p += np.dot(test_weights**2, np.diag(covariance_matrix))
    test_sd_p = np.sqrt(test_sd_p)
    #  avg_r_p = nΣi=1(w_i * avg_r_i)
    test_r_p = np.dot(test_weights, expected_returns) 

    plt.scatter(
        test_sd_p,
        test_r_p,
        color='blue',
        s=2,
    )
    plt.scatter(
        [portfolio.standard_deviation],
        [portfolio.expected_return],
        color='red',
        s=4,
    )
    plt.plot(sd_p, r_p)
    plt.xlabel("Portfolio Standard Deviation")
    plt.ylabel("Portfolio Return")
    plt.show()


def run(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
):
    weights = [0.1] * len(stocks)

    covariance_matrix = calculate_covariance_matrix(
        stocks,
        read_history_records,
    )

    # portfolio_expected_return = calculate_portfolio_expected_return(
    #     weights,
    #     stocks,
    # )
    # portfolio_standard_deviation = calculate_portfolio_standard_deviation(
    #     weights,
    #     stocks,
    #     read_history_records,
    #     covariance_matrix,
    # )
    # print(f'Expected Return of EW    {portfolio_expected_return}')
    # print(f'Standard Deviation of EW {portfolio_standard_deviation}')

    # print()

    # global_mv_portfolio = calculate_global_minimum_variance_portfolio(
    #     stocks,
    #     read_history_records,
    #     covariance_matrix,
    # )
    # print('Global minimum variance portfolio')
    # print(f'Portfolio variance           {global_mv_portfolio.variance}')
    # print(f'Portfolio standard deviation {global_mv_portfolio.standard_deviation}')
    # print(f'Portfolio expected return    {global_mv_portfolio.expected_return}')
    # print('Portfolio weights:')
    # for i, stock in enumerate(stocks):
    #     print(f'Stock {stock.code:>5} weight = {global_mv_portfolio.weights[i]}')

    calculate_efficient_frontier(stocks, read_history_records, covariance_matrix)
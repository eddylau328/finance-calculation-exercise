import itertools
from typing import Callable, List

import numpy as np
import math

from common import Stock, StockRecord
from part1 import calculate_stock_price_returns, calculate_annualized_standard_deviation, calculate_annualized_covariance

from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage.filters import gaussian_filter1d


def generate_dirichlet_sets(n, k):
    alpha = np.ones(k)  # Set alpha parameter to 1
    sets = np.random.dirichlet(alpha, n)
    return sets


def calculate_portfolio_expected_return(
    weights: List[float],
    stocks: List[Stock],
) -> float:
    expected_returns = [stock.expected_return for stock in stocks]
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
    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])
    covariance_matrix = np.cov(returns_matrix) * 252
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))


def calculate_global_minimum_variance_portfolio(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
) -> List[float]:
    num_assets = len(stocks)
    expected_returns = [stock.expected_return for stock in stocks]
    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])
    covariance_matrix = np.cov(returns_matrix) * 252

    def objective_function(weights): return np.sqrt(
        np.dot(weights.T, np.dot(covariance_matrix, weights)))

    bounds = [(-1, 1) for _ in range(len(stocks))]

    def equality_constraint(weights): return np.sum(weights) - 1
    # inequality_constraint = lambda weights: weights

    constraints = [
        {'type': 'eq', 'fun': equality_constraint},
    ]

    initial_weights = np.random.uniform(low=-0, high=1, size=10)
    initial_weights = initial_weights/np.sum(initial_weights)

    result = minimize(
        objective_function,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    optimal_sd = objective_function(result.x)
    print(f'Optimal SD {optimal_sd}')
    print(f'Optimal Weights {result.x.round(3)}')
    print(f'Expected Return {np.dot(expected_returns, np.array(result.x))}')


def draw_efficient_frontier(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
) -> List[float]:
    num_assets = len(stocks)
    expected_returns = [stock.expected_return for stock in stocks]

    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])
    covariance_matrix = np.cov(returns_matrix) * 252
    def objective_function(weights): return np.sqrt(
        np.dot(weights.T, np.dot(covariance_matrix, weights)))

    num_assets = len(stocks)
    num_portfolios = 10000

    p_weights = np.zeros((num_portfolios, num_assets))
    p_return = np.zeros(num_portfolios)
    p_sd = np.zeros(num_portfolios)

    weights_sets = generate_dirichlet_sets(num_portfolios, num_assets)

    for idx in range(num_portfolios):
        # weights =  np.random.uniform(low=0, high=1, size=10)
        # weights = weights/np.sum(weights)
        weights = weights_sets[idx]
        # Returns are the product of individual expected returns of asset and its
        returns = np.dot(weights, expected_returns)
        sd = objective_function(weights)

        p_weights[idx] = np.array(weights)
        p_return[idx] = returns
        p_sd[idx] = sd

    # -- Skip over dominated portfolios
    dominated_weights = []
    dominated_returns = []
    dominated_sd = []

    for i in range(num_portfolios):
        dominated = False
        for j in range(num_portfolios):
            if i != j:
                if p_return[j] - p_return[i] > 0 and p_sd[i] - p_sd[j] > 0:
                    dominated = True
                    break
        if not dominated:
            dominated_weights.append(p_weights[i])
            dominated_returns.append(p_return[i])
            dominated_sd.append(p_sd[i])

    # create a list of tuples pairing each dominated return with its corresponding dominated standard deviation
    dominated_data = list(
        zip(dominated_returns, dominated_sd, dominated_weights))

    # sort the dominated data by the second element of each tuple (i.e., by the standard deviation)
    sorted_dominated_data = sorted(dominated_data, key=lambda x: x[1])

    # extract the sorted returns and standard deviations into separate lists
    sorted_dominated_returns = [data[0] for data in sorted_dominated_data]
    sorted_dominated_sd = [data[1] for data in sorted_dominated_data]
    sorted_dominated_weights = [data[2] for data in sorted_dominated_data]

    dominated_returns = sorted_dominated_returns
    dominated_sd = sorted_dominated_sd

    smoothed_dominated_sd = gaussian_filter1d(dominated_sd, sigma=3)
    smoothed_dominated_returns = gaussian_filter1d(dominated_returns, sigma=3)

    # plot
    plt.figure(figsize=(10, 7))
    plt.scatter(p_sd, p_return, c=p_return/p_sd, marker='.', s=5)
    plt.plot(smoothed_dominated_sd, smoothed_dominated_returns, color="r")
    plt.grid(True)
    plt.xlabel('sd')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')

    # plt.show()

    return smoothed_dominated_sd, smoothed_dominated_returns, sorted_dominated_weights


def find_optimal_portfolio(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
    risk_free_rate: float,
):

    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])

    expected_returns = [stock.expected_return for stock in stocks]

    covariance_matrix = np.cov(returns_matrix) * 252

    def objective_function(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_sd = np.sqrt(
            np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_sd
        return -sharpe_ratio

    constraints = [
        {'type': 'ineq', 'fun': lambda weight: weight},
        {'type': 'eq', 'fun': lambda weight: np.sum(weight) - 1}
    ]

    bounds = [(0, 1) for _ in range(len(stocks))]

    initial_weights = np.random.uniform(low=0, high=1, size=10)

    result = minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds, options={
        'maxiter': 1000
    })

    return result.x.round(3)


def find_updated_optimal_portfolio(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
):
    risk_free_rate = 0.043

    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])

    expected_returns = [
        stock.analyst_update_expected_return for stock in stocks]

    covariance_matrix = np.cov(returns_matrix)

    def objective_function(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_sd = np.sqrt(
            np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_sd
        return -sharpe_ratio

    constraints = [
        {'type': 'ineq', 'fun': lambda weight: weight},
        {'type': 'eq', 'fun': lambda weight: np.sum(weight) - 1}
    ]

    bounds = [(0, 1) for _ in range(len(stocks))]

    initial_weights = np.random.uniform(low=0, high=1, size=10)

    result = minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds, options={
        'maxiter': 1000
    })

    return result.x.round(3)


def print_2_ii_a_get_erc_marginal_risk_contribution(weights,
                                                    stocks: List[Stock],
                                                    portfolio_standard_deviation,
                                                    read_history_records):
    """
        mr_i = ((wi*q_i**2)+(Σk<>i(wk*q_ik)))/q_p
    """

    for i in range(len(stocks)):
        stock_records_i = read_history_records(stocks[i].code)
        w_i = weights[i]
        q_i = calculate_annualized_standard_deviation(stock_records_i)
        numerator = (w_i * (q_i ** 2))
        for k in range(len(stocks)):
            if i != k:
                stock_records_k = read_history_records(stocks[k].code)
                q_ik = calculate_annualized_covariance(
                    stock_records_i,
                    stock_records_k,
                )
                numerator += weights[k] * q_ik
        mr_i = numerator / portfolio_standard_deviation
        print("marginal_risk of ", stocks[i].code, ": ", mr_i)


def run(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
    risk_free_rate: float,
):
    weights = [0.1] * len(stocks)

    portfolio_expected_return = calculate_portfolio_expected_return(
        weights,
        stocks
    )
    portfolio_standard_deviation = calculate_portfolio_standard_deviation(
        weights,
        stocks,
        read_history_records,
    )

    print(f'Expected Return of EW    {portfolio_expected_return}')
    print(f'Standard Deviation of EW {portfolio_standard_deviation}')

    # calculate_global_minimum_variance_portfolio(stocks, read_history_records)
    # returns_matrix = np.array([
    #     calculate_stock_price_returns(read_history_records(stock.code))
    #     for stock in stocks
    # ])
    # efficient_frontier = draw_efficient_frontier(stocks, read_history_records)
    # optimal_portfolio = find_optimal_portfolio(
    #     stocks,
    #     read_history_records,
    #     risk_free_rate,
    # )

    # print(f'Optimal Portfolio {optimal_portfolio}')
    # covariance_matrix = np.cov(returns_matrix) * 252

    # expected_returns = [stock.expected_return for stock in stocks]

    # updated_optimal_portfolio = find_updated_optimal_portfolio(stocks, read_history_records)
    # print(f'Updated Optimal Portfolio {updated_optimal_portfolio}')

    # p_sd = np.sqrt(np.dot(optimal_portfolio.T, np.dot(covariance_matrix, optimal_portfolio)))
    # p_return = np.dot(optimal_portfolio, expected_returns)

    # risk_free_rate = 0.043

    # # plt.figure(figsize=(10, 7))
    # # plt.plot(efficient_frontier[0], efficient_frontier[1], color="r")
    # # plt.plot(0, 0.043, 'o', color="black")
    # plt.plot(p_sd, p_return, color="black")
    # plt.plot([0, p_sd], [0.043, p_return], color="black")
    # plt.grid(True)
    # plt.xlabel('sd')
    # plt.ylabel('Expected Return')

    # plt.show()

import itertools
from typing import Callable, List

import numpy as np

from common import Stock, StockRecord
from part1 import calculate_stock_price_returns

from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage.filters import gaussian_filter1d


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
    covariance_matrix = np.cov(returns_matrix)
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
    covariance_matrix = np.cov(returns_matrix)

    objective_function = lambda weights: np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    bounds = [(-1, 1) for _ in range(len(stocks))]

    equality_constraint = lambda weights: np.sum(weights) - 1
    # inequality_constraint = lambda weights: weights

    constraints = [
        {'type': 'eq', 'fun': equality_constraint},
    ]

    initial_weights =  np.random.uniform(low=-0, high=1, size=10)
    initial_weights = initial_weights/np.sum(initial_weights)

    result = minimize(
        objective_function,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    optimal_sd =  objective_function(result.x)
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
    covariance_matrix = np.cov(returns_matrix)
    objective_function = lambda weights: np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    num_assets = len(stocks)
    num_portfolios = 10000

    p_weights = np.zeros((num_portfolios, num_assets))
    p_return = np.zeros(num_portfolios)
    p_sd = np.zeros(num_portfolios)

    for _ in range(num_portfolios):
        weights =  np.random.uniform(low=0, high=1, size=10)
        weights = weights/np.sum(weights)
        returns = np.dot(weights, expected_returns) # Returns are the product of individual expected returns of asset and its
        sd = objective_function(weights)

        p_weights[_] = np.array(weights)
        p_return[_] = returns
        p_sd[_] = sd

    #-- Skip over dominated portfolios
    dominated_returns = []
    dominated_sd = []

    for i in range(num_portfolios) :
        dominated = False
        for j in range(num_portfolios):
            if i != j:
                if p_return[j] - p_return[i] > 0 and p_sd[i] - p_sd[j] > 0:
                    dominated = True
                    break
        if not dominated:
            dominated_returns.append(p_return[i])
            dominated_sd.append(p_sd[i])

    # create a list of tuples pairing each dominated return with its corresponding dominated standard deviation
    dominated_data = list(zip(dominated_returns, dominated_sd))

    # sort the dominated data by the second element of each tuple (i.e., by the standard deviation)
    sorted_dominated_data = sorted(dominated_data, key=lambda x: x[1])

    # extract the sorted returns and standard deviations into separate lists
    sorted_dominated_returns = [data[0] for data in sorted_dominated_data]
    sorted_dominated_sd = [data[1] for data in sorted_dominated_data]

    dominated_returns = sorted_dominated_returns
    dominated_sd = sorted_dominated_sd

    smoothed_dominated_sd = gaussian_filter1d(dominated_sd, sigma=2)
    smoothed_dominated_returns = gaussian_filter1d(dominated_returns, sigma=2)

    # plot
    plt.figure(figsize=(10, 7))
    plt.scatter(p_sd, p_return, c=p_return/p_sd, marker='.')
    plt.plot(smoothed_dominated_sd, smoothed_dominated_returns, color="r")
    plt.grid(True)
    plt.xlabel('sd')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')

    # plt.show()

    return smoothed_dominated_sd, smoothed_dominated_returns


def find_optimal_portfolio(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
): 
    risk_free_rate = 0.043
    
    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])

    expected_returns = [stock.expected_return for stock in stocks]

    covariance_matrix = np.cov(returns_matrix)

    def objective_function(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_sd = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_sd
        return -sharpe_ratio

    constraints = [
        {'type': 'ineq', 'fun': lambda weight: weight},
        {'type': 'eq', 'fun': lambda weight: np.sum(weight) - 1}
    ]

    bounds = [(0, 1) for _ in range(len(stocks))]

    initial_weights =  np.random.uniform(low=0, high=1, size=10)

    result = minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)

    print(result)

    return result.x.round(3)

def run(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
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
    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])
    efficient_frontier = draw_efficient_frontier(stocks, read_history_records)
    optimal_portfolio = find_optimal_portfolio(stocks, read_history_records)
    covariance_matrix = np.cov(returns_matrix)

    print(optimal_portfolio.round(3))
    print(np.sum(optimal_portfolio.round(3)))

    expected_returns = [stock.expected_return for stock in stocks]
    
    p_sd = np.sqrt(np.dot(optimal_portfolio.T, np.dot(covariance_matrix, optimal_portfolio)))
    p_return = np.dot(optimal_portfolio, expected_returns)

    risk_free_rate = 0.043


    def cal(weights):
        risky_sd = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        portfolio_sd = np.dot(risky_sd, weights)
        risky_return = np.dot(expected_returns, weights)
        beta = (risky_return - risk_free_rate) / risky_sd
        return risk_free_rate + beta * portfolio_sd

    # plt.figure(figsize=(10, 7))
    # plt.plot(efficient_frontier[0], efficient_frontier[1], color="r")
    plt.plot(p_sd, p_return, 'o', color="black")
    plt.grid(True)
    plt.xlabel('sd')
    plt.ylabel('Expected Return')

    plt.show()



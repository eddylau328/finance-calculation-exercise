from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
import matplotlib.pyplot as plt

from part1 import calculate_stock_price_returns
import numpy as np


def part_b(
    stocks,
    read_history_records,
):
    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])
    expected_returns = [stock.expected_return for stock in stocks]
    cov_matrix = np.cov(returns_matrix) * 252
    ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(-1, 1))
    ef.min_volatility()
    ret_tangent, std_tangent, _ = ef.portfolio_performance()
    print("global minimum variance portfolio")
    print(f"standard deviation = {ret_tangent}")
    print(f"expect return      = {std_tangent}")
    print("weights")
    print(ef.weights)


def plot_efficient_frontier(
    expected_returns,
    cov_matrix,
    filename,
):
    ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0, 1))
    fig, ax = plt.subplots()
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Generate random portfolios
    n_samples = 20000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_xbound(0)
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def run_efficient_frontier(
    expected_returns,
    cov_matrix,
    risk_free_rate,
    filename,
    plot_title,
):
    ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0, 1))
    ef1 = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0, 1))
    # ef.min_volatility()
    # print(ef.weights)

    # fig, ax = plt.subplots()
    # plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
    # plt.show()

    fig, ax = plt.subplots()
    ef_max_sharpe = ef1
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find the tangency portfolio
    ef_max_sharpe.max_sharpe()
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()

    print("optimal portfolio")
    print(f"standard deviation = {std_tangent}")
    print(f"expected return    = {ret_tangent}")
    print(f"weights")
    print(ef_max_sharpe.weights)

    ax.scatter(std_tangent, ret_tangent, marker="*",
               s=100, c="r", label="Max Sharpe")

    # Generate random portfolios
    n_samples = 20000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # plot risk free asset
    ax.scatter(0, risk_free_rate, marker="x",
               s=100, c="r", label="Risk free")
    ax.plot(
        [
            0,
            std_tangent,
        ],
        [
            risk_free_rate,
            ret_tangent,
        ],
        c="black",
        label="CAL",
    )

    # Output
    ax.set_xbound(0)
    ax.set_title(plot_title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def part_c_and_d(
    stocks,
    read_history_records,
    risk_free_rate,
):
    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])
    expected_returns = [stock.expected_return for stock in stocks]
    cov_matrix = np.cov(returns_matrix) * 252
    plot_efficient_frontier(expected_returns, cov_matrix, "ef_scatter_c.png")
    run_efficient_frontier(
        expected_returns,
        cov_matrix,
        risk_free_rate,
        "ef_scatter_d.png",
        "Part D: Efficient Frontier with random portfolios",
    )


def part_e(
    stocks,
    read_history_records,
    risk_free_rate,
):
    print("Updated forecast!")
    returns_matrix = np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ])
    expected_returns = [
        stock.analyst_update_expected_return for stock in stocks
    ]
    cov_matrix = np.cov(returns_matrix) * 252
    run_efficient_frontier(
        expected_returns,
        cov_matrix,
        risk_free_rate,
        "ef_scatter_e.png",
        "Part E: Efficient Frontier with random portfolios",
    )


def run(
    stocks,
    read_history_records,
    risk_free_rate,
):
    part_b(stocks, read_history_records)
    print()
    part_c_and_d(stocks, read_history_records, risk_free_rate)
    print()
    part_e(stocks, read_history_records, risk_free_rate)

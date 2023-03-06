from typing import List
import riskfolio as rp
import numpy as np
import pandas as pd

from part1 import (
    calculate_stock_price_returns,
    calculate_annualized_standard_deviation,
    calculate_annualized_covariance,
)
from part2 import (
    calculate_portfolio_expected_return,
    calculate_portfolio_standard_deviation,
)
from common import Stock


def print_2_ii_a_get_erc_marginal_risk_contribution(
    weights,
    stocks: List[Stock],
    portfolio_standard_deviation,
    read_history_records
):
    """
        mr_i = ((wi*q_i**2)+(Σk<>i(wk*q_ik)))/q_p
    """
    mr = []
    tr = 0
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
        tr = w_i * mr_i
        print(f"marginal risk for {stocks[i].code} = {mr_i}")
        print(f"total risk from {stocks[i].code}   = {tr}")
        print()
        mr.append(mr_i)

    return mr, tr


def run(
    stocks,
    read_history_records,
    risk_free_rate,
):
    returns_matrix = pd.DataFrame(np.array([
        calculate_stock_price_returns(read_history_records(stock.code))
        for stock in stocks
    ]).T)
    port = rp.Portfolio(returns=returns_matrix)

    # port.assets_stats(method_mu='hist', method_cov='hist', d=0.94)
    # 计算期望收益及方差，当模型model选择Classic时，需使用assets_stats计算组合的期望收益及方差
    method_mu = 'hist'  # 还支持其他方法，详见文档
    method_cov = 'hist'  # 还支持其他方法，详见文档
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)

    w_rp = port.rp_optimization(
        model="Classic",  # use historical
        rm="MV",  # use mean-variance optimization
        hist=True,  # use historical scenarios
        rf=0,  # set risk free rate to 0
        b=None  # don't use constraints
    )

    print(w_rp)
    print()
    weights = list(w_rp.values.T[0])
    mr, tr = print_2_ii_a_get_erc_marginal_risk_contribution(
        weights,
        stocks,
        calculate_portfolio_standard_deviation(
            weights,
            stocks,
            read_history_records
        ),
        read_history_records,
    )

    ew_sd = calculate_portfolio_standard_deviation(
        [0.1] * len(stocks),
        stocks,
        read_history_records,
    )
    print(f'Standard Deviation of optimal ERC portfolio  = {tr}')
    print(f'Standard Deviation of Leverage ERC portfolio = {ew_sd}')
    leverage_factor = ew_sd/tr
    print(f'Leverage factor = {leverage_factor}')
    print()

    erc_expect_return = calculate_portfolio_expected_return(
        weights,
        stocks,
    )
    erc_sharp_ratio = (erc_expect_return - risk_free_rate) / tr
    print(f'Sharp ratio of optimal ERC portfolio = {erc_sharp_ratio}')
    leverage_expect_return = erc_sharp_ratio * ew_sd + risk_free_rate
    print(f'Leverage expect return = {leverage_expect_return}')

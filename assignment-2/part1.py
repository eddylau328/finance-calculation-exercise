import math
import random
from typing import List, Callable, Tuple
from string import Template

import numpy as np

from common import (
    Stock,
    StockRecord,
)

INTEREST_RATE = 0.0375


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


def part_1_1_i(
        stock_1_records: List[StockRecord],
        stock_2_records: List[StockRecord],
) -> Tuple[float, float]:
    stock_1_volatility = calculate_realized_volatility([r.price for r in stock_1_records])
    stock_2_volatitliy = calculate_realized_volatility([r.price for r in stock_2_records])
    return stock_1_volatility, stock_2_volatitliy


def print_part_1_1_i(results: Tuple[float, float]):
    template = Template('Stock S$index realized volatility = $result')
    print('Part1. 1i) Realized Volatilities')
    for i, result in enumerate(results):
        print(template.safe_substitute(index=i + 1, result=result))
    print()


def part_1_1_ii(
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
        stock_1_volatility: float,
        stock_2_volatility: float,
        corr_coef: float
):
    # stock_1 price at time=0
    s1_0 = stock_1_records[stock_2_records.__len__() - 1].price

    # stock_2 price at time=0
    s2_0 = stock_2_records[stock_2_records.__len__() - 1].price

    # number of times for each period
    times_2_i = int(252 / 2)
    time_steps_2_i = 180

    part_1_2_generate_path("Part1. 2i)(a)", s1_0, s2_0, time_steps_2_i, times_2_i, stock_1_volatility, stock_2_volatility,
                             corr_coef, 1000)
    part_1_2_generate_path("Part1. 2i)(b)", s1_0, s2_0, time_steps_2_i, times_2_i, stock_1_volatility, stock_2_volatility,
                             corr_coef, 10000)
    part_1_2_generate_path("Part1. 2i)(c)", s1_0, s2_0, time_steps_2_i, times_2_i, stock_1_volatility, stock_2_volatility,
                             corr_coef, 100000)
    part_1_2_generate_path("Part1. 2i)(d)", s1_0, s2_0, time_steps_2_i, times_2_i, stock_1_volatility, stock_2_volatility,
                             corr_coef, 500000)

    print()

    # number of times for each period
    times_2_ii = 1
    time_steps_2_ii = 2

    part_1_2_generate_path("Part1. 2ii)(a)", s1_0, s2_0, time_steps_2_ii, times_2_ii, stock_1_volatility, stock_2_volatility,
                             corr_coef, 1000)
    part_1_2_generate_path("Part1. 2ii)(b)", s1_0, s2_0, time_steps_2_ii, times_2_ii, stock_1_volatility, stock_2_volatility,
                             corr_coef, 10000)
    part_1_2_generate_path("Part1. 2ii)(c)", s1_0, s2_0, time_steps_2_ii, times_2_ii, stock_1_volatility, stock_2_volatility,
                             corr_coef, 100000)
    part_1_2_generate_path("Part1. 2ii)(d)", s1_0, s2_0, time_steps_2_ii, times_2_ii, stock_1_volatility, stock_2_volatility,
                             corr_coef, 500000)


def part_1_2_generate_path(
        print_tag: str,
        s1_0: float,
        s2_0: float,
        time_steps: int,
        times: int,
        stock_1_volatility: float,
        stock_2_volatility: float,
        corr_coef: float,
        num_of_paths: int
):
    sum_a = 0
    for i in range(num_of_paths):
        epsilon_1 = random.uniform(-1, 1)
        # TODO epsilon_2 should be obtained by epsilon_1
        # epsilon_2 = corr_coef*x_1+x_2*math.sqrt(1-corr_coef^2) , dont know how to find x_1 and x_2???
        epsilon_2 = random.uniform(-1, 1)
        s1_1 = calculate_single_path(first_day_stock_price=s1_0,
                                     epsilon=epsilon_1,
                                     time_steps=time_steps,
                                     times=times,
                                     volatility=stock_1_volatility)
        s2_1 = calculate_single_path(first_day_stock_price=s2_0,
                                     epsilon=epsilon_2,
                                     time_steps=time_steps,
                                     times=times,
                                     volatility=stock_2_volatility)
        s1_2 = calculate_single_path(first_day_stock_price=s1_1,
                                     epsilon=epsilon_1,
                                     time_steps=time_steps,
                                     times=times,
                                     volatility=stock_1_volatility)
        s2_2 = calculate_single_path(first_day_stock_price=s2_1,
                                     epsilon=epsilon_2,
                                     time_steps=time_steps,
                                     times=times,
                                     volatility=stock_2_volatility)
        b1 = max(s1_1 / s1_0, s2_1 / s2_0)
        b2 = max(s1_2 / s1_0, s2_2 / s2_0)
        a = (b1 + b2) / 2
        sum_a += a
        # print(f'==================Path{i}==================')
        # print('==================')
        # print(f's1_0 = {s1_0}')
        # print(f's1_1 = {s1_1}')
        # print(f's1_2 = {s1_2}')
        # print('==================')
        # print(f's2_0 = {s2_0}')
        # print(f's2_1 = {s2_1}')
        # print(f's2_2 = {s2_2}')
    print(f'{print_tag} A for number of paths = {num_of_paths} : {sum_a / num_of_paths}')



def calculate_single_path(first_day_stock_price: float, epsilon: float, time_steps: int, times: int,
                          volatility: float) -> float:
    stock_price = first_day_stock_price
    delta_t = 1 / time_steps
    # print(f'first_day_stock_price = {first_day_stock_price}')
    for i in range(times):
        delta_s = INTEREST_RATE * delta_t + volatility * epsilon * math.sqrt(delta_t)
        stock_price += delta_s
        # print(f'{i} delta_s = {delta_s}, new_stock_price = {stock_price}')
    return stock_price


def print_part_1_1_ii(result: float):
    print('Part1. 1ii) Correlation Coefficient')
    print(f'correlation coef. of S1 and S2 = {result}')
    print()


def run(
        stock_1: Stock,
        stock_2: Stock,
        read_stock_data: Callable[[int], List[StockRecord]],
):
    stock_1_records = read_stock_data(stock_1.code)
    stock_2_records = read_stock_data(stock_2.code)

    volatilities = part_1_1_i(stock_1_records, stock_2_records)
    print_part_1_1_i(volatilities)

    corr_coef = part_1_1_ii(stock_1_records, stock_2_records)
    print_part_1_1_ii(corr_coef)

    part_1_2_i(stock_1_records, stock_2_records, volatilities[0], volatilities[1], corr_coef)

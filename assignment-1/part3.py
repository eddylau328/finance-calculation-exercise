from typing import Callable, List
from common import Stock, StockRecord

import pandas as pd
import numpy as np
import yfinance as yf
import os
import part2_alt
import part2_ii


path = './part_3_records.csv'


def print_performance(
    codes,
    stocks,
    initial_capital,
    columns,
    weights,
    leverage_factor=None,
):
    total_gain = 0
    for i, code in enumerate(codes[0:len(stocks)]):
        data = columns[code]
        percentage_change = (data[data.shape[0]-1] - data[0]) / data[0]
        if not leverage_factor:
            stock_gain = percentage_change * weights[i] * initial_capital
        else:
            stock_gain = percentage_change * \
                weights[i] * initial_capital * leverage_factor
        print(f'Gain of {code} = {stock_gain}')
        total_gain += stock_gain
    print(f'Total gain = {total_gain}')
    print(f'Total      = {initial_capital + total_gain}')


def run(
    stocks: List[Stock],
    read_history_records: Callable[[int], List[StockRecord]],
    risk_free_rate: float,
):
    start = "2023-01-31"
    end = "2023-02-21"
    codes = [
        str(stock.code).zfill(4)+'.HK' for stock in stocks
    ] + ['^HSI']

    if not os.path.isfile(path):
        data = yf.download(codes, start=start, end=end)
        data.to_csv('./part_3_records.csv')

    data = pd.read_csv(path)
    columns = {}
    for i in range(len(stocks)+1):
        column_name = 'Close'
        if i > 0:
            column_name += f'.{i}'
        columns[data[column_name][0]
                ] = data[column_name].values[2:].astype(np.float64)

    initial_capital = 1000000.0
    data = columns['^HSI']
    hsi_percentage_change = (data[data.shape[0]-1] - data[0]) / data[0]
    print()
    print(f'HSI percentage change = {hsi_percentage_change}')
    print()

    # part 2 i a
    print('Part 2 i a')
    weights = np.array([0.1] * len(stocks))
    print_performance(
        codes,
        stocks,
        initial_capital,
        columns,
        weights,
    )
    print()

    print('Part 2 i b')
    weights = part2_alt.part_b(stocks, read_history_records, True)
    print_performance(
        codes,
        stocks,
        initial_capital,
        columns,
        weights,
    )
    print()

    print('Part 2 i d')
    weights = part2_alt.part_c_and_d(
        stocks,
        read_history_records,
        risk_free_rate,
        True
    )
    print_performance(
        codes,
        stocks,
        initial_capital,
        columns,
        weights,
    )
    print()

    print('Part 2 ii a')
    weights, leverage_factor = part2_ii.run_silent(
        stocks,
        read_history_records,
        risk_free_rate,
    )
    print_performance(
        codes,
        stocks,
        initial_capital,
        columns,
        weights,
    )
    print()

    print('Part 2 ii b')
    print_performance(
        codes,
        stocks,
        initial_capital,
        columns,
        weights,
        leverage_factor,
    )
    print()

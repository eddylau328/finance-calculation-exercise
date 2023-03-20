from typing import List, Tuple, Callable
from string import Template
import argparse

import pandas as pd

from common import (
    Stock,
    StockDetail,
    StockRecord,
    Group,
)
import part1
import part2


LAST_DIGIT = 9
STOCK_DATA_FILE_PATH = './assignment 2 stock data.xlsx'
IMPLIED_VOLATILITY_SHEET_NAME = 'implied volatility'
STOCK_PRICE_SHEET_NAME = 'stock prices'


def setup() -> Tuple[
    Stock,
    Stock,
    StockDetail,
    Callable[[int], List[StockRecord]],
]:
    def read_stock_data(stock_code: int) -> List[StockRecord]:
        df = pd.read_excel(
            STOCK_DATA_FILE_PATH,
            sheet_name=STOCK_PRICE_SHEET_NAME,
            skiprows=4,
            usecols='C:W',
        )
        df.rename({'Stock code': 'record_date'}, axis=1, inplace=True)
        if stock_code not in list(df.columns.values)[1:]:
            raise Exception(f'Stock code {stock_code} not in datasheet')
        return [
            StockRecord(
                date=df['record_date'][i],
                price=df[stock_code][i],
            ) for i in range(len(df.index))
        ]

    def read_part_2_group(group_num: int) -> Group:
        df = pd.read_excel(
            STOCK_DATA_FILE_PATH,
            sheet_name=IMPLIED_VOLATILITY_SHEET_NAME,
            skiprows=18,
            nrows=5,
            usecols='J:M',
            names=[
                'attribute',
                'group_1',
                'group_2',
                'group_3',
            ],
        )
        if not (1 <= group_num <= 3):
            raise Exception('Invalid group number is provided')
        column = f'group_{group_num}'
        # Assuming the ordering stay the same in the excel file
        # strike: float
        # knock_in_level: float
        # auto_call_level: float
        # coupon_per_month_set_1: float
        # coupon_per_month_set_2: float
        return Group(*list(df[column].values))
    
    def read_part_1_volatility() -> Tuple[Stock, Stock]:
        df = pd.read_excel(
            STOCK_DATA_FILE_PATH,
            sheet_name=IMPLIED_VOLATILITY_SHEET_NAME,
            skiprows=4,
            nrows=10,
            usecols="C:G",
            names=[
                'order',
                'stock_code_1',
                'stock_name_1',
                'stock_code_2',
                'stock_name_2',
            ],
        )
        index = LAST_DIGIT - 1
        return (
            Stock(code=df['stock_code_1'][index], name=df['stock_name_1'][index]),
            Stock(code=df['stock_code_2'][index], name=df['stock_name_2'][index]),
        )

    def read_part_2_volatility() -> Stock:
        df = pd.read_excel(
            STOCK_DATA_FILE_PATH,
            sheet_name=IMPLIED_VOLATILITY_SHEET_NAME,
            skiprows=18,
            nrows=10,
            usecols="C:H",
            names=[
                'order',
                'stock_code',
                'stock_name',
                'initial_stock_price',
                'implied_volatility',
                'group',
            ],
        )
        index = LAST_DIGIT - 1
        group = read_part_2_group(df['group'][index])
        return StockDetail(
            code=df['stock_code'][index],
            name=df['stock_name'][index],
            initial_stock_price=df['initial_stock_price'][index],
            implied_volatility=df['implied_volatility'][index],
            group=group,
        )

    part_1_stock_1, part_1_stock_2 = read_part_1_volatility()
    part_2_stock = read_part_2_volatility()
    return part_1_stock_1, part_1_stock_2, part_2_stock, read_stock_data


def main(args: List[str]):
    part_1_stock_1, part_1_stock_2, part_2_stock, read_stock_data = setup()
    select_stock_template = Template(
        '$syntax Stock: $code $name'
    )
    if 'part1' in args:
        print(f'Last digits is {LAST_DIGIT}, we use these stocks for Part 1')
        print(select_stock_template.safe_substitute(
                syntax='S1',
                code=f'{str(part_1_stock_1.code):<5}',
                name=part_1_stock_1.name,
            )
        )
        print(select_stock_template.safe_substitute(
                syntax='S2',
                code=f'{str(part_1_stock_2.code):<5}',
                name=part_1_stock_2.name,
            )
        )
        part1.run(
            part_1_stock_1,
            part_1_stock_2,
            read_stock_data,
        )
        print()
    
    if 'part2' in args:
        print(f'Last digits is {LAST_DIGIT}, we use these stocks for Part 2')
        print(select_stock_template.safe_substitute(
                syntax='S1',
                code=f'{str(part_2_stock.code):<5}',
                name=part_2_stock.name,
            )
        )
        part2.run(part_2_stock, read_stock_data)
        print()


if __name__ == '__main__':
    run_parts = ['part1', 'part2']
    parser = argparse.ArgumentParser(description='CMSC5718 Assignment 2')
    parser.add_argument('part_arg', type=str, default=run_parts, nargs='*',
                        help=f'An optional string for running specific parts, i.e. {run_parts}')
    args = parser.parse_args()
    if not all([p in run_parts for p in args.part_arg]):
        raise Exception(f'Invalid arguments {args.part_arg}')

    main(args.part_arg)

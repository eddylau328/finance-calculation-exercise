import argparse
from typing import Callable, List, Tuple

import pandas as pd

from common import Stock, StockRecord
import part1
import part2


STOCK_DATA_FILE_PATH = './assignment 1 stock data.xlsx'
EXPECTED_RETURN_SHEET_NAME = 'Consensus Expected return'
STOCK_PRICE_SHEET_NAME = 'stock prices'


def setup() -> Tuple[List[Stock], Callable[[int], List[StockRecord]], float]:

    def read_stocks() -> List[Stock]:
        df = pd.read_excel(STOCK_DATA_FILE_PATH,
                           sheet_name=EXPECTED_RETURN_SHEET_NAME)
        # get 10 stocks from the sheet
        # sheet does not have named columns
        return [
            Stock(
                code=int(df['Unnamed: 2'][2+i]),
                name=str(df['Unnamed: 3'][2+i]),
                expected_return=float(df['Unnamed: 4'][2+i]),
                analyst_update_expected_return=float(df['Unnamed: 6'][2+i]),
            ) for i in range(0, 10)
        ]

    def read_history_records(code: int) -> List[StockRecord]:
        df = pd.read_excel(
            STOCK_DATA_FILE_PATH,
            sheet_name=STOCK_PRICE_SHEET_NAME,
            index_col=1,
            skiprows=4,
        )
        if code not in df.keys():
            raise Exception(f'Invalid {code} provided')
        # start from 1, as need to skip empty row
        # assume data are sorted
        return [
            StockRecord(
                date=df.index[i],
                price=df[code][i],
            ) for i in range(1, len(df[code]))
        ]

    def read_risk_free_rate() -> float:
        df = pd.read_excel(STOCK_DATA_FILE_PATH,
                           sheet_name=EXPECTED_RETURN_SHEET_NAME)
        risk_free_rate = df['Unnamed: 4'][13]
        if type(risk_free_rate) != float:
            raise Exception('risk free rate is not defined')
        return risk_free_rate

    stocks = read_stocks()
    risk_free_rate = read_risk_free_rate()
    return stocks, read_history_records, risk_free_rate


def main(parts: List[str]):
    stocks, read_history_records, risk_free_rate = setup()

    if 'part1' in parts:
        print("Part 1")
        part1.run(
            stocks,
            read_history_records,
        )
        print()

    if 'part2' in parts:
        print("Part 2")
        part2.run(
            stocks,
            read_history_records,
            risk_free_rate,
        )
        print()


if __name__ == '__main__':
    run_parts = ['part1', 'part2', 'part3']
    parser = argparse.ArgumentParser(description='CMSC5718 Assignment 1')
    parser.add_argument('part_arg', type=str, default=run_parts, nargs='*',
                        help=f'An optional string for running specific parts, i.e. {run_parts}')
    args = parser.parse_args()
    if not all([p in run_parts for p in args.part_arg]):
        raise Exception(f'Invalid arguments {args.part_arg}')

    main(args.part_arg)

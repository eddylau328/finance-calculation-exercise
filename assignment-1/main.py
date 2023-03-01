from typing import Callable, List, Tuple
import pandas as pd

from common import Stock, StockRecord
import part1

STOCK_DATA_FILE_PATH = './assignment 1 stock data.xlsx'
EXPECTED_RETURN_SHEET_NAME = 'Consensus Expected return'
STOCK_PRICE_SHEET_NAME = 'stock prices'


def setup() -> Tuple[List[Stock], Callable[[int], List[StockRecord]]]:

    def read_stocks() -> List[Stock]:
        df = pd.read_excel(STOCK_DATA_FILE_PATH, sheet_name=EXPECTED_RETURN_SHEET_NAME)
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

    stocks = read_stocks()
    return stocks, read_history_records 


def main():
    stocks, read_history_records = setup()
    part1.run(
        stocks,
        read_history_records,
    )
    

if __name__ == '__main__':
    main()

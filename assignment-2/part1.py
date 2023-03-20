from typing import List, Callable

from common import (
    Stock,
    StockRecord,
)

def run(
    stock_1: Stock,
    stock_2: Stock,
    read_stock_data: Callable[[int], List[StockRecord]],
):
    pass
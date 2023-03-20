from typing import List, Callable

from common import (
    StockDetail,
    StockRecord,
)

def run(
    stock: StockDetail,
    read_stock_data: Callable[[int], List[StockRecord]],
):
    pass
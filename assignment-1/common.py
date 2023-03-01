from datetime import datetime
from dataclasses import dataclass


@dataclass
class Stock:
    """ Stock data extract from Consensus Expected Return Sheet
    code:                           stock code
    name:                           stock name
    expectet_return:                expected 1-yr return
    analyst_update_expected_return: expected 1-yr return with analyst update
    """
    code: int
    name: str
    expected_return: float
    analyst_update_expected_return: float


@dataclass
class StockRecord:
    """ Stock historic record
    price: stock price
    date:  record date
    """
    date: datetime
    price: float

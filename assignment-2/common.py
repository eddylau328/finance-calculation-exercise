from dataclasses import dataclass
from datetime import datetime


@dataclass
class StockRecord:
    """ Stock historic record
    price: stock price
    date:  record date
    """
    date: datetime
    price: float


@dataclass
class Stock:
    """ Stock data extract from implied volatility Sheet
    code:                           stock code
    name:                           stock name
    """
    code: int
    name: str


@dataclass
class Group:
    """
    strike:                         Strike % K0
    knock_in_level:                 Knock-in level KI
    auto_call_level:                Auto-call level AC
    coupon_per_month_set_1:         Coupon per month CP, set 1
    coupon_per_month_set_2:         Coupon per month CP, set 2
    """
    strike: float
    knock_in_level: float
    auto_call_level: float
    coupon_per_month_set_1: float
    coupon_per_month_set_2: float


@dataclass
class StockDetail:
    """ Stock detail data extract from implied volatility Sheet
    code:                           stock code
    name:                           stock name
    initial_stock_price             stock price
    implied_volatility
    group
    """
    code: int
    name: str
    initial_stock_price: float
    implied_volatility: float
    group: Group
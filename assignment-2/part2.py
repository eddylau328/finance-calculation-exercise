import math
import random
from typing import List, Callable
import numpy as np

from common import (
    StockDetail,
    StockRecord,
)

INTEREST_RATE = 0.0375
NOMINAL_AMOUNT = 100000


def part2_1_1and2(stock: StockDetail):
    fair_price1000_set1 = generate_fair_price(print_tag="part2) 1a)",
                                              num_of_paths=1000,
                                              stock=stock,
                                              cp_rate_per_month=stock.group.coupon_per_month_set_1)
    print(f"Profit of investment bank = {NOMINAL_AMOUNT - fair_price1000_set1}")

    fair_price10000_set1 = generate_fair_price(print_tag="part2) 1b)",
                                               num_of_paths=10000,
                                               stock=stock,
                                               cp_rate_per_month=stock.group.coupon_per_month_set_1)
    print(f"Profit of investment bank = {NOMINAL_AMOUNT - fair_price10000_set1}")

    fair_price100000_set1 = generate_fair_price(print_tag="part2) 1c)",
                                                num_of_paths=100000,
                                                stock=stock,
                                                cp_rate_per_month=stock.group.coupon_per_month_set_1)
    print(f"Profit of investment bank = {NOMINAL_AMOUNT - fair_price100000_set1}")

    fair_price500000_set1 = generate_fair_price(print_tag="part2) 1c)",
                                                num_of_paths=500000,
                                                stock=stock,
                                                cp_rate_per_month=stock.group.coupon_per_month_set_1)
    print(f"Profit of investment bank = {NOMINAL_AMOUNT - fair_price500000_set1}")

    fair_price1000_set2 = generate_fair_price(print_tag="part2) 2a)",
                                              num_of_paths=1000,
                                              stock=stock,
                                              cp_rate_per_month=stock.group.coupon_per_month_set_2)
    print(f"Profit of investment bank = {NOMINAL_AMOUNT - fair_price1000_set2}")
    print(f"Additional profit of the investment bank = {fair_price1000_set1-fair_price1000_set2}")

    fair_price10000_set2 = generate_fair_price(print_tag="part2) 2b)",
                                               num_of_paths=10000,
                                               stock=stock,
                                               cp_rate_per_month=stock.group.coupon_per_month_set_2)
    print(f"Profit of investment bank = {NOMINAL_AMOUNT - fair_price10000_set2}")
    print(f"Additional profit of the investment bank = {fair_price10000_set1 - fair_price10000_set2}")

    fair_price100000_set2 = generate_fair_price(print_tag="part2) 2c)",
                                                num_of_paths=100000,
                                                stock=stock,
                                                cp_rate_per_month=stock.group.coupon_per_month_set_2)
    print(f"Profit of investment bank = {NOMINAL_AMOUNT - fair_price100000_set2}")
    print(f"Additional profit of the investment bank = {fair_price100000_set1 - fair_price100000_set2}")

    fair_price500000_set2 = generate_fair_price(print_tag="part2) 2d)",
                                                num_of_paths=500000,
                                                stock=stock,
                                                cp_rate_per_month=stock.group.coupon_per_month_set_2)
    print(f"Profit of investment bank = {NOMINAL_AMOUNT - fair_price500000_set2}")
    print(f"Additional profit of the investment bank = {fair_price500000_set1 - fair_price500000_set2}")


def part2_1_3(stock: StockDetail):
    cp_rate_per_month= 1.5 / 100
    fair_price = generate_fair_price(print_tag="part2) 3)",
                        num_of_paths=500000,
                        stock=stock,
                        cp_rate_per_month=cp_rate_per_month)
    profit_margin = ((NOMINAL_AMOUNT-fair_price)/NOMINAL_AMOUNT) * 100
    print(f"Profit margin of the investment bank = {np.round(profit_margin, 4)}% when the CP level = {cp_rate_per_month*100}% ")


def generate_fair_price(print_tag: str,
                        num_of_paths: int,
                        stock: StockDetail,
                        cp_rate_per_month: float
                        ):
    nominal = NOMINAL_AMOUNT
    time_steps = 210  # Assume 1year = 360days
    volatility = stock.implied_volatility
    auto_call_price = stock.initial_stock_price * stock.group.auto_call_level
    first_auto_call_times = 30
    strike_price = stock.initial_stock_price * stock.group.strike
    knock_in_price = stock.initial_stock_price * stock.group.knock_in_level
    cp_interval = 30

    total_return = 0
    auto_call_count = 0
    for i in range(num_of_paths):
        path_result = calculate_single_path(stock.initial_stock_price, nominal, time_steps, volatility, auto_call_price,
                                            first_auto_call_times, strike_price, knock_in_price, cp_interval,
                                            cp_rate_per_month)
        is_auto_call = path_result[0]
        value = path_result[1]
        total_return += value
        # print(f"{i} auto call = {is_auto_call}, value ={value}")
        if is_auto_call:
            auto_call_count += 1
    fair_price = total_return / num_of_paths
    print()
    print(f"{print_tag}")
    print(f"Number of paths = {num_of_paths}\nAuto call count = {auto_call_count}\nFair price = {fair_price}")
    return fair_price


# return [0] isAutoCall:Boolean - True if early terminated, otherwise False
# return [1] value_at_the_end:float - The total value get back from investor
def calculate_single_path(
    initial_stock_price: float,
    nominal: float,
    time_steps: int,
    volatility: float,
    auto_call_price: float,
    first_auto_call_times: int,
    strike_price: float,
    knock_in_price: float,
    cp_interval: int,
    cp_rate_per_month: float
):
    stock_price = initial_stock_price
    coupon_earn = 0
    delta_t = 1 / time_steps
    knock_in_occurred = False
    last_coupon_time = 0
    for i in range(1, time_steps + 1):
        epsilon = np.random.normal()
        delta_s = stock_price * (INTEREST_RATE * delta_t + volatility * epsilon * math.sqrt(delta_t))
        stock_price += delta_s
        # print(f'{i} delta_s = {delta_s}, new_stock_price = {stock_price}')

        # knock-in event
        if stock_price < knock_in_price:
            knock_in_occurred = True

        # early terminal return
        if i >= first_auto_call_times and stock_price >= auto_call_price:
            return True, (nominal * (1 + ((i - last_coupon_time)/cp_interval) * cp_rate_per_month)) + coupon_earn
        
        # coupon paid to investor check
        if i % cp_interval == 0:
            coupon_earn += nominal * cp_rate_per_month  # coupon pay
            last_coupon_time = i
            # print(f'{i} coupon_earn = {coupon_earn}')

    if not knock_in_occurred or stock_price >= strike_price:
        return False, nominal + coupon_earn
    else:
        return False, nominal * (stock_price / strike_price) + coupon_earn
        # return nominal*(1+(early_terminal_time-last_coupon_time)*cp_rate_per_month)


def run(
    stock: StockDetail,
    read_stock_data: Callable[[int], List[StockRecord]],
):
    part2_1_1and2(stock=stock)
    part2_1_3(stock=stock)

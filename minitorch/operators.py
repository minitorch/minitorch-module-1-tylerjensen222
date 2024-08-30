"""
Collection of the core mathematical operators used throughout the code base.
"""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x, y):
    return x * y


# - id
def id(x):
    return x


# - add
def add(x, y):
    return x + y


# - neg


def neg(x):
    return -x


# - lt
def lt(x, y):
    return x < y


# - eq
def eq(x, y):
    return x == y


# - max
def max(x, y):
    if lt(x, y):
        return y
    else:
        return x


# - is_close
def is_close(x, y):
    return lt(abs(add(x, neg(y))), 1e-2)


# - sigmoid
def sigmoid(x):
    return 1.0 / add(1.0, exp(neg(x))) if lt(0, x) else exp(x) / add(1.0, exp(x))


# - relu
def relu(x):
    return max(x, 0)


# - log
def log(x):
    return math.log(x)


# - exp
def exp(x):
    return math.exp(x)


# - log_back
def log_back(x, y):
    return y / x


# - inv
def inv(x):
    return 1 / x


# - inv_back
def inv_back(x, d):
    return d * inv(x)


# - relu_back
def relu_back(x, d):
    if lt(0, x):
        return d
    else:
        return 0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(fn, iterable):
    return (fn(item) for item in iterable)


# - zipWith
def zipWith(fn, ls1: list[float], ls2: list[float]):
    return [fn(item1, item2) for item1, item2 in zip(ls1, ls2)]


# - reduce
def reduce(fn, ls: list[float]):
    value = ls[0]
    for element in ls[1:]:
        value = fn(value, element)
    return value


# Use these to implement
# - negList : negate a list
def negList(ls: list[float]):
    return map(neg, ls)


# - addLists : add two lists together
def addLists(ls1: list[float], ls2: list[float]):
    return zipWith(add, ls1, ls2)


# - sum: sum lists
def sum(ls: list[float]):
    if eq(len(ls), 0):
        return 0.0
    return reduce(add, ls)


# - prod: take the product of lists
def prod(ls: list[float]) -> float:
    return reduce(mul, ls)

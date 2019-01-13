"""
Simple Newton-Raphson Solver
"""

import numpy as np


def f(x: float) -> float:
    """ Function we'll find the (unique) zero of. """
    return np.exp(x) - np.exp(2)


def dfdx(x: float) -> float:
    """ Derivative of f """
    return np.exp(x)


def newton():
    """ Find where f(x) == 0 """
    x = 0
    max_iters = 100

    for i in range(max_iters):
        print(f'Iter #{i} - Guessing {x}')
        g = x - f(x) / dfdx(x)
        if abs(g - x) < 1e-24:
            break
        x = g

    if i == max_iters - 1:
        raise Exception


def main():
    newton()


if __name__ == '__main__':
    main()

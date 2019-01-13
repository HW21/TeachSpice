"""
Solver for a single, very special diode-resistor circuit.
"""

import numpy as np


class Component(object):
    pass


class Diode(Component):
    def __init__(self, isat=1e-16, vt=25e-3):
        self.isat = isat
        self.vt = vt

    def i(self, v):
        return self.isat * (np.exp(v / self.vt) - 1)

    def g(self, v):
        return (self.isat / self.vt) * np.exp(v / self.vt)

    def i0(self, v):
        return self.i(v) - self.g(v) * v


class Resistor(Component):
    pass


class Circuit:
    pass


class Solver:
    pass


G = 1e-3
V = 1.0
D = Diode()


def converge(vk: float, vkm1: float) -> bool:
    """ Convergence test """
    abs_tol = 1e-18
    return abs(vk - vkm1) < abs_tol


def next_guess(v: float) -> float:
    """ Update method for Newton iterations """
    f = D.i0(v) + (D.g(v) * v) + (G * v) - (G * V)
    df = D.g(v) + G
    return v - f / df


def diode_res_ckt():
    vg = 1.0
    iters = 100

    while iters:
        print(f'Guessing {vg}')
        vgn = next_guess(vg)
        if converge(vgn, vg):
            break
        vg = vgn
        iters = iters - 1

    if not iters:
        raise Exception

    print(f'Diode Current: {D.i(vgn)}')
    print(f'Res   Current: {G*(V-vgn)}')


def main():
    diode_res_ckt()


if __name__ == '__main__':
    main()

"""
(V - Vd)G = Id(Vd)

f(Vd) = Id(Vd) + G* Vd - G*V
f_lin = i0(Vd) + gd(Vd) + G*Vd - G*V
 
df/dVd = dId/dVd + G
df_lin/dVd = gd + G 

Vk+1 = Vk - f(Vk) / df/dVd(Vk)
     = Vk - (Id(Vk) + G*Vk - G*V) / (gd + G)
     = Vk - (i0(Vk) + gd(Vk)*Vk + G*Vk - G*V) / (gd + G)
"""

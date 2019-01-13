"""
Solver for a single-variable circuit with arbitrary non-linear elements.
"""

import numpy as np


class Component(object):
    """ Base-class for (potentially non-linear) two-terminal elements. """

    def i(self, v: float) -> float:
        """ Returns the element current as a function of voltage `v`. """
        raise NotImplementedError

    def di_dv(self, v: float) -> float:
        """ Returns derivative of current as a function of voltage `v`. """
        raise NotImplementedError

    def limit(self, v: float) -> float:
        """ Voltage limits
        Base-class does no limiting.
        Sub-classes may implement bounds-checks here.
        Not this function must satisfy abs(limit(v)) <= abs(v), for any v. """
        return v


class Diode(Component):
    def __init__(self, isat=1e-16, vt=25e-3):
        super().__init__()
        self.isat = isat
        self.vt = vt

    def i(self, v: float) -> float:
        return -1 * self.isat * (np.exp(v / self.vt) - 1)

    def di_dv(self, v: float) -> float:
        return -1 * (self.isat / self.vt) * np.exp(v / self.vt)

    def limit(self, v):
        v = min(v, 1.5)
        v = max(v, -1.5)
        return v


class Resistor(Component):
    def __init__(self, r: float):
        self.r = r
        self.g = 1 / self.r

    def i(self, v: float) -> float:
        return -1 * self.g * v

    def di_dv(self, v: float) -> float:
        return -1 * self.g


class Isrc(Component):
    def __init__(self, idc: float):
        self.idc = idc

    def i(self, v: float) -> float:
        return self.idc

    def di_dv(self, v: float) -> float:
        return 0


class Node(object):
    """ The one and only Node in our circuit.
    Represents its connected elements, and knows how to perform KCL and its derivative. """

    def __init__(self):
        self.conns = []

    def add_conn(self, conn: Component):
        assert isinstance(conn, Component)
        self.conns.append(conn)

    def i(self, v):
        return sum([c.i(v) for c in self.conns])

    def di_dv(self, v):
        return sum([c.di_dv(v) for c in self.conns])

    def limit(self, v: float) -> float:
        for c in self.conns:
            v = c.limit(v)
        return v


class Solver:
    """ Single-Node Newton-Raphson Solver """

    def __init__(self, node: Node):
        self.node = node
        self.history = [0]

    @property
    def v(self):
        """ Our most-recent guess """
        return self.history[-1]

    def iterate(self) -> None:
        """ Update method for Newton iterations """
        f = self.node.i(self.v)
        df = self.node.di_dv(self.v)
        vn = self.v - f / df
        vl = self.node.limit(vn)
        self.history.append(vl)

    def converged(self) -> bool:
        """ Convergence test, including Newton-iteration-similarity and KCL. """
        v_tol = 1e-9
        if abs(self.history[-1] - self.history[-2]) >= v_tol:
            return False
        i_tol = 1e-9
        return abs(self.node.i(self.history[-1])) < i_tol

    def solve(self) -> None:
        """ Compute the Newton-Raphson-based non-linear solution. """

        max_iters = 100

        for i in range(max_iters):
            print(f'Iter #{i} - Guessing {self.v}')
            self.iterate()
            if self.converged():
                break

        if i >= max_iters - 1:
            raise Exception(f'Could Not Converge to Solution ')

        print(f'Successfully Converged to {self.v} in {i} iterations')


def main():
    """ Run a single-Node, multi-element simulation. """
    n = Node()

    r = Resistor(r=1e3)
    n.add_conn(r)
    i = Isrc(idc=1.5e-3)
    n.add_conn(i)
    d = Diode()
    n.add_conn(d)
    d = Diode(isat=1e-17)
    n.add_conn(d)

    s = Solver(node=n)
    s.solve()


if __name__ == '__main__':
    main()

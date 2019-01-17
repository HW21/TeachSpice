"""
Non-linear SPICE-style DC solver
"""

import numpy as np


class Component(object):
    """ Base-class for (potentially non-linear) two-terminal elements. """

    linear = False

    def __init__(self, *, p, n):
        self.p = p
        self.n = n
        self.p.add_conn(self)
        self.n.add_conn(self)

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
    def __init__(self, *, isat=1e-16, vt=25e-3, **kw):
        super().__init__(**kw)
        self.isat = isat
        self.vt = vt

    def i(self, v: float) -> float:
        v = self.limit(v)
        return 1 * self.isat * (np.exp(v / self.vt) - 1)

    def di_dv(self, v: float) -> float:
        v = self.limit(v)
        return 1 * (self.isat / self.vt) * np.exp(v / self.vt)

    def limit(self, v):
        v = min(v, 1.5)
        v = max(v, -1.5)
        return v


class Resistor(Component):
    linear = True

    def __init__(self, *, r: float, **kw):
        super().__init__(**kw)
        self.r = r
        self.g = 1 / self.r


class Isrc(Component):
    linear = True

    def __init__(self, *, idc: float, **kw):
        super().__init__(**kw)
        self.idc = idc


class Node(object):
    """ The one and only Node in our circuit.
    Represents its connected elements, and knows how to perform KCL and its derivative. """

    def __init__(self):
        self.conns = []
        self.num = None

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


class Circuit(object):
    """ Circuit Under Test for Simulation. """

    def __init__(self):
        self.node0 = Node()
        self.nodes = [self.node0]
        self.comps = []

    def create_nodes(self, num: int):
        for k in range(num):
            self.add_node(Node())

    def add_node(self, node: Node) -> int:
        assert isinstance(node, Node)
        node.num = len(self.nodes)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def add_comp(self, comp: Component) -> Component:
        assert isinstance(comp, Component)
        assert comp.p in self.nodes
        assert comp.n in self.nodes
        self.comps.append(comp)
        return comp


class Solver:
    """ Newton-Raphson Solver

    f(x) = G*x + H*g(x) - s
    Jf(x) * dx + f(x) = 0
    Jf(x) = df(x)/dx = G + Jg(x)
    """

    def __init__(self, ckt: Circuit):
        self.ckt = ckt
        ckt.solver = self

        self.num_nodes = len(ckt.nodes) - 1
        self.G = np.zeros((self.num_nodes, self.num_nodes))
        self.Jg = np.zeros((self.num_nodes, self.num_nodes))
        self.Hg = np.zeros(self.num_nodes)
        self.s = np.zeros(self.num_nodes)
        self.x = np.zeros(self.num_nodes)
        self.history = [self.x]

        for comp in self.ckt.comps:
            if not comp.linear:
                # Deal with non-linear components elsewhere!
                continue
            if isinstance(comp, Resistor):
                if comp.p is not self.ckt.node0:
                    self.G[comp.p.num - 1, comp.p.num - 1] += comp.g
                if comp.n is not self.ckt.node0:
                    self.G[comp.n.num - 1, comp.n.num - 1] += comp.g
                if comp.p is not self.ckt.node0 and comp.n is not self.ckt.node0:
                    self.G[comp.p.num - 1, comp.n.num - 1] -= comp.g
                    self.G[comp.n.num - 1, comp.p.num - 1] -= comp.g
            elif isinstance(comp, Isrc):
                self.s[comp.p.num - 1] = 1 * comp.idc
            else:
                raise NotImplementedError(f'Unknown Component {comp}')

        print(self.G)
        print(self.s)

    def update(self):
        """ Update non-linear component operating points """
        self.Jg = np.zeros((self.num_nodes, self.num_nodes))
        self.Hg = np.zeros(self.num_nodes)
        for comp in self.ckt.comps:
            if comp.linear:
                continue
            if isinstance(comp, Diode):
                vp = 0.0 if comp.p is self.ckt.node0 else self.x[comp.p.num - 1]
                vn = 0.0 if comp.n is self.ckt.node0 else self.x[comp.n.num - 1]
                v = vp - vn
                i = comp.i(v)
                di_dv = comp.di_dv(v)
                if comp.p is not self.ckt.node0:
                    self.Hg[comp.p.num - 1] += i
                    self.Jg[comp.p.num - 1, comp.p.num - 1] += di_dv
                if comp.n is not self.ckt.node0:
                    self.Hg[comp.n.num - 1] += i
                    self.Jg[comp.p.num - 1, comp.p.num - 1] += di_dv
                if comp.p is not self.ckt.node0 and comp.n is not self.ckt.node0:
                    self.Jg[comp.p.num - 1, comp.n.num - 1] -= di_dv
                    self.Jg[comp.n.num - 1, comp.p.num - 1] -= di_dv
            else:
                raise NotImplementedError(f'Unknown Component {comp}')

    def iterate(self) -> None:
        """ Update method for Newton iterations """
        # Update non-linear component operating points
        self.update()
        # Solve Jf(x) * dx + f(x) = 0
        rhs = -1 * (self.G.dot(self.x) + self.Hg - self.s)
        dx = np.linalg.solve(self.G + self.Jg, rhs)
        # print(f'Updating by {dx}')
        self.x += dx
        # vl = self.node.limit(vn)
        self.history.append(self.x)

    def converged(self) -> bool:
        """ Convergence test, including Newton-iteration-similarity and KCL. """
        self.update()
        v_tol = 1e-9
        for k in range(self.num_nodes):
            if abs(self.history[-1][k] - self.history[-2][k]) >= v_tol:
                return False
        i_tol = 1e-9
        i = self.G.dot(self.x) + self.Hg - self.s
        for k in range(self.num_nodes):
            if abs(i[k]) >= i_tol:
                return False
        return True

    def solve(self) -> None:
        """ Compute the Newton-Raphson-based non-linear solution. """

        max_iters = 100

        for i in range(max_iters):
            print(f'Iter #{i} - Guessing {self.x}')
            self.iterate()
            if self.converged():
                break

        if i >= max_iters - 1:
            raise Exception(f'Could Not Converge to Solution ')

        print(f'Successfully Converged to {self.x} in {i+1} iterations')


def main():
    ckt = Circuit()
    num_nodes = 50
    ckt.create_nodes(num_nodes)

    for k in range(num_nodes):
        r = Resistor(p=ckt.nodes[k + 1], n=ckt.nodes[k], r=(num_nodes - 1) * 1e3)
        ckt.add_comp(r)

    i = Isrc(p=ckt.nodes[num_nodes], n=ckt.node0, idc=1e-3 / (num_nodes - 1))
    d = Diode(p=ckt.nodes[1], n=ckt.node0)
    for _ in i, d:
        ckt.add_comp(_)

    s = Solver(ckt=ckt)
    s.solve()


if __name__ == '__main__':
    main()

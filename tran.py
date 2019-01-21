"""
Transient Solver

i = C * dV/dt
v(t) = integ(i/C, t) + v(0)
v(k+1) - v(k) = i(k+1) * dt / C

| 1 -dt/C | * | v(k+1) | = | v(k) |
              | i(k+1) |

or

i(k+1) = (v(k+1) - v(k)) * C/dt
       = v(k+1) * C/dt - v(k) * C/dt
       = v(k+1) * G - I
"""

import numpy as np

the_timestep = 1e-12


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


class Capacitor(Component):
    linear = True

    def __init__(self, *, c: float, **kw):
        super().__init__(**kw)
        self.c = c


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
        self.define()

    def define(self) -> None:
        """ Sub-classes can add content here. """
        pass

    def create_nodes(self, num: int):
        for k in range(num):
            self.add_node(Node())

    def add_node(self, node: Node) -> int:
        assert isinstance(node, Node)
        node.num = len(self.nodes)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def create_comp(self, cls: type, p: Node, n: Node, **kw) -> Component:
        comp = cls(p=p, n=n, **kw)
        return self.add_comp(comp=comp)

    def add_comp(self, comp: Component) -> Component:
        assert isinstance(comp, Component)
        assert comp.p in self.nodes
        assert comp.n in self.nodes
        self.comps.append(comp)
        return comp


class MatrixEq:
    """ Represents the non-linear matrix-equation
    G*x + H*g(x) = s
    """

    def __init__(self, ckt: Circuit):
        self.ckt = ckt
        ckt.mx = self

        self.num_nodes = len(ckt.nodes) - 1

        self.G = np.zeros((self.num_nodes, self.num_nodes))
        self.Gt = np.zeros((self.num_nodes, self.num_nodes))
        self.Jg = np.zeros((self.num_nodes, self.num_nodes))
        self.Hg = np.zeros(self.num_nodes)
        self.s = np.zeros(self.num_nodes)
        self.st = np.zeros(self.num_nodes)

        for comp in self.ckt.comps:
            if not comp.linear:
                # Deal with non-linear components elsewhere!
                continue
            elif isinstance(comp, Capacitor):
                continue
            elif isinstance(comp, Resistor):
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

    def update(self, x):
        """ Update non-linear component operating points """
        self.Jg = np.zeros((self.num_nodes, self.num_nodes))
        self.Hg = np.zeros(self.num_nodes)
        for comp in self.ckt.comps:
            if comp.linear:
                continue
            if isinstance(comp, Diode):
                vp = 0.0 if comp.p is self.ckt.node0 else x[comp.p.num - 1]
                vn = 0.0 if comp.n is self.ckt.node0 else x[comp.n.num - 1]
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

    def res(self, x):
        """ Return the residual error, given solution `x`. """
        return (self.G + self.Gt).dot(x) + self.Hg - self.s - self.st

    def solve(self, x):
        """ Solve our temporary-valued matrix for a change in x. """
        rhs = -1 * ((self.G + self.Gt).dot(x) + self.Hg - self.s - self.st)
        return np.linalg.solve(self.G + self.Gt + self.Jg, rhs)


class Solver:
    """ Newton-Raphson Solver

    f(x) = G*x + H*g(x) - s
    Jf(x) * dx + f(x) = 0
    Jf(x) = df(x)/dx = G + Jg(x)
    """

    def __init__(self, mx: MatrixEq, x0=None):
        self.mx = mx
        self.x = x0 or np.zeros(mx.num_nodes)
        self.history = [self.x]

    def update(self):
        """ Update non-linear component operating points """
        return self.mx.update(self.x)

    def iterate(self) -> None:
        """ Update method for Newton iterations """
        # Update non-linear component operating points
        self.update()
        # Solve Jf(x) * dx + f(x) = 0
        dx = self.mx.solve(self.x)
        # print(f'Updating by {dx}')
        self.x += dx
        # vl = self.node.limit(vn)
        self.history.append(self.x)

    def converged(self) -> bool:
        """ Convergence test, including Newton-iteration-similarity and KCL. """
        self.update()

        # Newton iteration similarity
        v_tol = 1e-9
        v_diff = self.history[-1] - self.history[-2]
        if np.any(np.abs(v_diff) >= v_tol):
            return False

        # KCL Residual
        i_tol = 1e-9
        i_res = self.mx.res(self.x)
        if np.any(i_res >= i_tol):
            return False

        # If we get here, they all passed!
        return True

    def solve(self) -> np.ndarray:
        """ Compute the Newton-Raphson-based non-linear solution. """
        max_iters = 100

        for i in range(max_iters):
            # print(f'Iter #{i} - Guessing {self.x}')
            self.iterate()
            if self.converged():
                break

        if i >= max_iters - 1:
            raise Exception(f'Could Not Converge to Solution ')

        # print(f'Successfully Converged to {self.x} in {i+1} iterations')
        return self.x


class Tran(object):
    """ Transient Solver """

    def __init__(self, mx: MatrixEq):
        self.mx = mx
        self.ckt = mx.ckt
        self.v = np.zeros(mx.num_nodes)
        self.history = [self.v]

    def update(self):
        """ Update time-dependent (dynamic) circuit element terms. """
        self.mx.Gt = np.zeros((self.mx.num_nodes, self.mx.num_nodes))
        self.mx.st = np.zeros(self.mx.num_nodes)

        for comp in self.ckt.comps:
            if isinstance(comp, Capacitor):
                # Update the "resistive" part of the cap, e.g. C/dt
                if comp.p is not self.ckt.node0:
                    self.mx.Gt[comp.p.num - 1, comp.p.num - 1] += comp.c / the_timestep
                if comp.n is not self.ckt.node0:
                    self.mx.Gt[comp.n.num - 1, comp.n.num - 1] += comp.c / the_timestep
                if comp.p is not self.ckt.node0 and comp.n is not self.ckt.node0:
                    self.mx.Gt[comp.p.num - 1, comp.n.num - 1] -= comp.c / the_timestep
                    self.mx.Gt[comp.n.num - 1, comp.p.num - 1] -= comp.c / the_timestep

                # Update the "past current" part of the cap, e.g. V[k-1]*C/dt
                vp = 0.0 if comp.p is self.ckt.node0 else self.v[comp.p.num - 1]
                vn = 0.0 if comp.n is self.ckt.node0 else self.v[comp.n.num - 1]
                v = vp - vn
                rhs = - 1 * comp.c * v / the_timestep
                if comp.p is not self.ckt.node0:
                    self.mx.st[comp.p.num - 1] = -1 * rhs
                if comp.n is not self.ckt.node0:
                    self.mx.st[comp.n.num - 1] = 1 * rhs

    def iterate(self):
        """ Run a single Newton solver """
        s = Solver(mx=self.mx)
        v = s.solve()
        # print(f'New Solution {v}')
        self.v = v
        self.history.append(self.v)

    def solve(self):
        t = 0

        for k in range(10000):
            t += the_timestep
            self.update()
            self.iterate()


def sim(c=1e-12):
    """ Create an instance of the circuit under test, and simulate it. """
    ckt = Circuit()
    num_nodes = 5
    ckt.create_nodes(num_nodes)

    for k in range(num_nodes):
        ckt.create_comp(cls=Resistor, p=ckt.nodes[k + 1], n=ckt.nodes[k], r=(num_nodes - 1) * 1e3)

    ckt.create_comp(cls=Isrc, p=ckt.nodes[num_nodes], n=ckt.node0, idc=1e-3 / (num_nodes - 1))
    ckt.create_comp(cls=Capacitor, p=ckt.nodes[1], n=ckt.node0, c=c)
    # ckt.create_comp(cls=Diode, p=ckt.nodes[1], n=ckt.node0)

    m = MatrixEq(ckt=ckt)
    s = Tran(mx=m)
    s.solve()

    return s


def main():
    import pandas as pd
    import matplotlib.pyplot as plt

    for c in 1e-14, 1e-13, 1e-12:
        s = sim(c=c)
        df = pd.DataFrame.from_records(s.history)
        print(df)
        plt.plot(df[0])
        plt.plot(df[1])
    plt.show()


if __name__ == '__main__':
    main()

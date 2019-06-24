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
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, AnyStr, SupportsFloat

the_timestep = 1e-12


class Component(object):
    """ Base-class for (potentially non-linear) two-terminal elements. """

    linear = False
    ports = []

    def __init__(self, *, conns: dict):
        assert list(conns.keys()) == self.ports
        self.ckt = None
        self.conns = conns
        for name, node in conns.items():
            node.add_conn(self)

    def get_v(self, x) -> Dict[AnyStr, SupportsFloat]:
        """ Get a dictionary of port-voltages, of the form `port_name`:`voltage`. """
        v = {}
        for name, node in self.conns.items():
            if node.num == 0:
                v[name] = 0.0
            else:
                v[name] = x[node.num - 1]
        return v

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

    def mna_setup(self, an) -> None:
        pass

    def mna_update(self, an) -> None:
        pass

    def tstep(self, an) -> None:
        pass

    def newton_iter(self, an) -> None:
        pass


# Port names for common two-terminal components
TWO_TERM_PORTS = ['p', 'n']


def mna_update_nonlinear_twoterm(comp, an):
    x = an.solver.x
    p = comp.conns['p']
    n = comp.conns['n']
    vp = 0.0 if p.num == 0 else x[p.num - 1]
    vn = 0.0 if n.num == 0 else x[n.num - 1]
    v = vp - vn
    if isinstance(comp, Diode):
        i = comp.i(v)
        di_dv = comp.di_dv(v)
    elif isinstance(comp, Capacitor):
        # Update the "resistive" part of the cap, e.g. C/dt, and
        # the "past current" part of the cap, e.g. V[k-1]*C/dt
        di_dv = comp.dq_dv(v) / the_timestep
        i = comp.q(v) / the_timestep - v * di_dv
    if p.num != 0:
        an.mx.Hg[p.num - 1] += i
        an.mx.Jg[p.num - 1, p.num - 1] += di_dv
    if n.num != 0:
        an.mx.Hg[n.num - 1] += i
        an.mx.Jg[p.num - 1, p.num - 1] += di_dv
    if p.num != 0 and n.num != 0:
        an.mx.Jg[p.num - 1, n.num - 1] -= di_dv
        an.mx.Jg[n.num - 1, p.num - 1] -= di_dv


class Diode(Component):
    ports = TWO_TERM_PORTS

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

    def mna_update(self, an) -> None:
        return mna_update_nonlinear_twoterm(self, an)


class Resistor(Component):
    ports = TWO_TERM_PORTS
    linear = True

    def __init__(self, *, r: float, **kw):
        super().__init__(**kw)
        self.r = r
        self.g = 1 / self.r

    def mna_setup(self, an) -> None:
        p = self.conns['p']
        n = self.conns['n']
        if p.num != 0:
            an.mx.G[p.num - 1, p.num - 1] += self.g
        if n.num != 0:
            an.mx.G[n.num - 1, n.num - 1] += self.g
        if p.num != 0 and n.num != 0:
            an.mx.G[p.num - 1, n.num - 1] -= self.g
            an.mx.G[n.num - 1, p.num - 1] -= self.g


class Capacitor(Component):
    ports = TWO_TERM_PORTS
    linear = False

    def __init__(self, *, c: float, **kw):
        super().__init__(**kw)
        self.c = c

    def q(self, v: float) -> float:
        return self.c * v

    def dq_dv(self, v: float) -> float:
        return self.c

    def tstep(self, an) -> None:
        p = self.conns['p'].num
        n = self.conns['n'].num
        v = self.get_v(an.v)
        vd = v['p'] - v['n']
        # Update the "past charge" part of the cap equation
        rhs = - 1 * self.q(vd) / the_timestep
        if p != 0:
            an.mx.st[p - 1] = -1 * rhs
        if n != 0:
            an.mx.st[n - 1] = 1 * rhs

    def mna_update(self, an) -> None:
        return mna_update_nonlinear_twoterm(self, an)


class Isrc(Component):
    """ Constant Current Source """

    ports = TWO_TERM_PORTS
    linear = True

    def __init__(self, *, idc: float, **kw):
        super().__init__(**kw)
        self.idc = idc

    def mna_setup(self, an):
        s = an.mx.s
        p = self.conns['p']
        s[p.num - 1] = 1 * self.idc


class Bjt(Component):
    ports = ['c', 'b', 'e']


class Mos(Component):
    """ Level-Zero MOS Model """
    ports = ['g', 'd', 's', 'b']
    vth = 0.2
    beta = 1e-6
    lam = 0  # Sorry "lambda" is a keyword

    def mna_update(self, an) -> None:
        mx = an.mx
        v = self.get_v(an.v)
        # i = self.i(v)

        vds = v['d'] - v['s']
        vgs = v['g'] - v['s']
        vov = vgs - self.vth
        if vov <= 0:  # Cutoff
            ids = 0
            gm = 0
            gds = 0
        elif vds >= vov:  # Saturation
            ids = self.beta / 2 * (vov ** 2) * (1 + self.lam * vds)
            gm = self.beta * vov * (1 + self.lam * vds)
            gds = self.lam * self.beta / 2 * (vov ** 2)
        else:  # Triode
            ids = self.beta * ((vov ** vds) - (vds ** 2) / 2)
            gm = self.beta * vds
            gds = self.beta * (vov - vds)
        # return dict(d=ids, g=0, s=-1 * ids, b=0)

        dn = self.conns['d'].num
        sn = self.conns['s'].num
        gn = self.conns['g'].num
        assert self.conns['b'].num == 0  # No floating bulk, yet
        if dn != 0:
            mx.Hg[dn - 1] -= ids
            mx.Jg[dn - 1, dn - 1] -= gds
        if sn != 0:
            mx.Hg[sn - 1] += ids
            mx.Jg[sn - 1, sn - 1] += gm + gds
        if dn != 0 and sn != 0:
            mx.Jg[dn - 1, sn - 1] -= (gm + gds)
            mx.Jg[sn - 1, dn - 1] -= gds
        if gn != 0 and sn != 0:
            mx.Jg[sn - 1, gn - 1] -= gm
        if gn != 0 and dn != 0:
            mx.Jg[dn - 1, gn - 1] += gm

    def di_dv(self, v: np.ndarray) -> np.ndarray:
        """ Returns a 4x4 matrix of current-derivatives """
        # FIXME!
        return np.zeros((4, 4))


class Node(object):
    """ Circuit Node """

    def __init__(self):
        self.conns = []
        self.num = None
        self.ckt = None

    def add_conn(self, conn: Component):
        assert isinstance(conn, Component)
        self.conns.append(conn)

    def limit(self, v: float) -> float:
        for c in self.conns:
            v = c.limit(v)
        return v


class Circuit(object):
    """ Circuit Under Test for Simulation. """

    def __init__(self):
        self.comps = []
        self.nodes = []
        # Create node zero, and initialize our node-list
        self.node0 = Node()
        self.add_node(self.node0)
        # Run any custom definition
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
        node.ckt = self
        self.nodes.append(node)
        return len(self.nodes) - 1

    def create_comp(self, *, cls: type, **kw) -> Component:
        """ Create and add Component of class `cls`, and return it. """
        comp = cls(**kw)
        comp.ckt = self
        return self.add_comp(comp=comp)

    def add_comp(self, comp: Component) -> Component:
        """ Add Component `comp`, and return it. """
        assert isinstance(comp, Component)
        for name, node in comp.conns.items():
            assert node in self.nodes
        self.comps.append(comp)
        return comp


class MnaSystem(object):
    """
    Represents the non-linear matrix-equation
    G*x + H*g(x) = s

    And the attendant break-downs which help us solve it.
    f(x) = G*x + H*g(x) - s
    Jf(x) * dx + f(x) = 0
    Jf(x) = df(x)/dx = G + Jg(x)
    """

    def __init__(self, ckt, an):
        self.ckt = ckt
        self.an = an
        ckt.mx = self

        self.num_nodes = len(ckt.nodes) - 1

        self.G = np.zeros((self.num_nodes, self.num_nodes))
        self.Gt = np.zeros((self.num_nodes, self.num_nodes))
        self.Jg = np.zeros((self.num_nodes, self.num_nodes))
        self.Hg = np.zeros(self.num_nodes)
        self.s = np.zeros(self.num_nodes)
        self.st = np.zeros(self.num_nodes)

    def update(self, x: np.ndarray) -> None:
        """ Update non-linear component operating points """
        self.Jg = np.zeros((self.num_nodes, self.num_nodes))
        self.Hg = np.zeros(self.num_nodes)
        for comp in self.ckt.comps:
            comp.mna_update(self.an)

    def res(self, x: np.ndarray) -> np.ndarray:
        """ Return the residual error, given solution `x`. """
        return (self.G + self.Gt).dot(x) + self.Hg - self.s - self.st

    def solve(self, x: np.ndarray) -> np.ndarray:
        """ Solve our temporary-valued matrix for a change in x. """
        rhs = -1 * self.res(x)
        dx = np.linalg.solve(self.G + self.Gt + self.Jg, rhs)
        return dx


class Solver:
    """ Newton-Raphson Solver """

    def __init__(self, mx: MnaSystem, x0=None):
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
        self.x += dx
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


class Analysis(object):
    def __init__(self, ckt: Circuit, **kw):
        super().__init__(**kw)
        self.ckt = ckt
        self.mx = MnaSystem(ckt=ckt, an=self)
        for comp in self.ckt.comps:
            comp.mna_setup(self)


class DcOp(Analysis):
    pass


class Tran(Analysis):
    """ Transient Solver """

    def __init__(self, ckt: Circuit):
        super().__init__(ckt)
        self.v = np.zeros(self.mx.num_nodes)
        self.history = [self.v]
        self.solver = None

    def update(self):
        """ Update time-dependent (dynamic) circuit element terms. """
        self.mx.Gt = np.zeros((self.mx.num_nodes, self.mx.num_nodes))
        self.mx.st = np.zeros(self.mx.num_nodes)
        for comp in self.ckt.comps:
            comp.tstep(an=self)

    def iterate(self):
        """ Run a single Newton solver """
        self.solver = Solver(mx=self.mx)
        v = self.solver.solve()
        # print(f'New Solution {v}')
        self.v = v
        self.history.append(self.v)

    def solve(self):
        t = 0
        for k in range(10000):
            t += the_timestep
            self.update()
            self.iterate()


def rc_filter(c=1e-12, num_nodes=5):
    class RcFilter(Circuit):
        """ Current-Driven RC Filter, 
        with Parameterizable R, C, and # intermediate nodes """

        def define(self):
            self.create_nodes(num_nodes)

            for k in range(num_nodes):
                self.create_comp(cls=Resistor,
                                 r=(num_nodes - 1) * 1e3,
                                 conns=dict(
                                     p=self.nodes[k + 1],
                                     n=self.nodes[k],
                                 ), )

            self.create_comp(cls=Isrc,
                             idc=1e-3 / (num_nodes - 1),
                             conns=dict(
                                 p=self.nodes[num_nodes],
                                 n=self.node0,
                             ), )

            self.create_comp(cls=Capacitor,
                             c=c,
                             conns=dict(
                                 p=self.nodes[1],
                                 n=self.node0,
                             ), )
            # self.create_comp(cls=Diode, p=self.nodes[1], n=self.node0)

    return RcFilter()


def sim(c=1e-12):
    """ Create an instance of the circuit under test, and simulate it. """
    num_nodes = 5
    ckt = rc_filter(c=c, num_nodes=num_nodes)

    s = Tran(ckt=ckt)
    s.solve()

    return s


def main():
    """ Run some sims!
    Sweep and plot a few RC parameters. """

    def add_tran_plots(sim):
        df = pd.DataFrame.from_records(sim.history)
        # print(df)
        plt.plot(df[0])
        plt.plot(df[1])

    for c in 1e-14, 1e-13, 1e-12:
        s = sim(c=c)
        add_tran_plots(s)

    plt.show()


if __name__ == '__main__':
    main()

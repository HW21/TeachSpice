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

from . import the_timestep
from .circuit import Circuit
from .solve import MnaSystem, Solver
from .components import Resistor


class Analysis(object):
    def __init__(self, ckt: Circuit, **kw):
        super().__init__(**kw)
        self.ckt = ckt
        self.solver = None
        self.mx = None

    def mna_setup(self):
        self.mx = MnaSystem(ckt=self.ckt, an=self)
        for comp in self.ckt.comps:
            comp.mna_setup(self)


class DcOp(Analysis):
    def solve(self, x0):
        """ Solve for steady-state response.
        Includes gmin-stepping style homotopy. """

        # Set up gmin resistors
        rmin_exponent = 0
        rmin_resistors = []
        for node in self.ckt.nodes:
            r = Resistor(r=10 ** rmin_exponent, conns={'p': node, 'n': self.ckt.node0})
            self.ckt.add_comp(r)
            rmin_resistors.append(r)
        self.mna_setup()

        xi = np.copy(x0)
        while rmin_exponent <= 12:
            print(f'Solving with rmin=10**{rmin_exponent}')
            self.solver = Solver(mx=self.mx, x0=xi)
            xi = self.solver.solve()
            print(f'Solution: {xi}')
            rmin_exponent += 1
            for r in rmin_resistors:
                r.update(r=10 ** rmin_exponent)
            self.mna_setup()

        return self.v

    @property
    def v(self):
        return self.solver.x


class Tran(Analysis):
    """ Transient Solver """

    def __init__(self, ckt: Circuit):
        super().__init__(ckt)
        self.mna_setup()
        self.v = np.zeros(self.mx.num_nodes)
        self.history = [self.v]

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

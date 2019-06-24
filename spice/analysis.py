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
from .circuit import Circuit, Node
from .solve import MnaSystem, Solver


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

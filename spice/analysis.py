"""
Analysis Classes
"""

import numpy as np

from . import the_timestep
from .circuit import Circuit
from .solve import MnaSystem, TheSolverCls
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

    @property
    def v(self):
        return self.solver.x


class DcOp(Analysis):
    def solve(self, x0=[0.0]):
        """ Solve for steady-state response.
        Includes gmin-stepping style homotopy. """

        x = np.array(x0, dtype='float64') if np.any(x0) else np.zeros(len(self.ckt.nodes))

        # Set up gmin resistors
        rmin_exponent = 0
        rmin_resistors = []
        for num, node in enumerate(self.ckt.nodes):
            # Gmin resistors tie to initial-condition guesses. Create them if necessary.
            if x[num] == 0:
                n = self.ckt.node0
            else:
                # FIXME: can re-use nodes of same voltage-value
                n = self.ckt.create_forced_node(name=f'node{num}_force', v=x[num])
            r = Resistor(r=10 ** rmin_exponent, conns={'p': node, 'n': n})
            self.ckt.add_comp(r)
            rmin_resistors.append(r)
        self.mna_setup()

        xi = np.copy(x0)
        while rmin_exponent <= 12:
            # print(f'Solving with rmin=10**{rmin_exponent}')

            self.solver = TheSolverCls(self, x0=xi)

            xi = self.solver.solve()
            # print(f'Solution: {xi}')
            rmin_exponent += 1
            for r in rmin_resistors:
                r.update(r=10 ** rmin_exponent)
            self.mna_setup()

        return self.v


class Tran(Analysis):
    """ Transient Solver

    Notes, mostly on caps:

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

    def __init__(self, ckt: Circuit):
        super().__init__(ckt)
        self.mna_setup()
        self.history = []
        self.t = 0.0

    def update(self):
        """ Update time-dependent (dynamic) circuit element terms. """
        self.mx.Gt = np.zeros((self.mx.size, self.mx.size))
        self.mx.st = np.zeros(self.mx.size)
        for comp in self.ckt.comps:
            comp.tstep(an=self)

    def iterate(self):
        """ Run a single Newton solver """
        # FIXME: initialize this initial-guess better
        if self.solver:
            x0 = self.v
        else:
            x0 = None
        self.solver = TheSolverCls(self, x0=x0)
        self.update()
        try:
            v = self.solver.solve()
            self.history.append(v)
        except Exception as e:
            print(f'Failed to converge at {self.t}: {e}')
            raise e

    def solve(self):
        for k in range(1000):
            self.iterate()
            self.t += the_timestep


class Contour(Analysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xs = []
        self.ys = []
        self.dxs = []
        self.xfs = []

    @property
    def v(self):
        return self.solver.x

    def explore(self, *, xmin=-1.0, xmax=1.0, xstep=0.1):
        import itertools

        nstep = int((xmax - xmin) / xstep) + 1
        dim = np.linspace(xmin, xmax, nstep)
        grid = len(self.ckt.nodes) * [dim]

        for xi in itertools.product(*grid):
            xi = np.array(xi)

            self.mna_setup()
            self.solver = TheSolverCls(self, x0=xi)
            self.solver.update()
            y = self.solver.mx.res(self.solver.x)
            dx = self.solver.mx.solve(self.solver.x)

            try:
                self.solver = TheSolverCls(self, x0=xi)
                xf = self.solver.solve()
            except:
                xf = len(self.ckt.nodes) * [np.NaN]

            self.xs.append(xi)
            self.ys.append(y)
            self.dxs.append(dx)
            self.xfs.append(xf)

        return self.xs, self.ys, self.dxs, self.xfs

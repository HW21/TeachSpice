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

from typing import Dict, AnyStr, SupportsFloat
import numpy as np
import scipy.optimize

from . import the_timestep
from .circuit import Circuit
from .solve import MnaSystem, Solver
from .components import Resistor


class ScipySolver:
    """ Solver based on scipy.optimize.minimize
    The general idea is *minimizing* KCL error, rather than solving for when it equals zero.
    To our knowledge, no other SPICE solver tries this.  Maybe it's a bad idea. """

    def __init__(self, an, *, x0=None):
        self.an = an
        self.x0 = np.array(x0, dtype='float64') if np.any(x0) else np.zeros(len(an.ckt.nodes))
        self.x = self.x0
        self.history = [self.x0]
        self.results = []

    def solve(self):
        options = dict(fatol=1e-31, disp=False)
        result = scipy.optimize.minimize(fun=self.guess, x0=self.x0, method='nelder-mead', options=options)

        if not result.success:
            raise TabError(str(result))
        print(f'Solved: {result.x}')
        return result.x

    def get_v(self, comp) -> Dict[AnyStr, SupportsFloat]:
        """ Get a dictionary of port-voltages, of the form `port_name`:`voltage`. """
        v = {}
        for name, node in comp.conns.items():
            if node.solve:
                v[name] = self.x[node.num]
            else:
                v[name] = self.an.ckt.forces[node]
        return v

    def guess(self, x):
        # print(f'Guessing {x}')
        self.x = x
        self.history.append(x)
        an = self.an
        kcl_results = np.zeros(len(an.ckt.nodes))
        for comp in an.ckt.comps:
            comp_v = self.get_v(comp)
            # print(f'{comp} has voltages {comp_v}')
            comp_i = comp.i(comp_v)
            # {d:1.3, s:=1.3, g:0, b:0}
            for name, i in comp_i.items():
                node = comp.conns[name]
                if node.solve:
                    # print(f'{comp} updating {node} by {i}')
                    kcl_results[node.num] += i
        # print(f'KCL: {kcl_results}')
        rv = np.sum(kcl_results ** 2)
        self.results.append(rv)
        return rv


""" 'Configuration' of which Solver to use """
TheSolverCls = Solver
# TheSolverCls = ScipySolver


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
        self.t = 0.0

    def update(self):
        """ Update time-dependent (dynamic) circuit element terms. """
        self.mx.Gt = np.zeros((self.mx.num_nodes, self.mx.num_nodes))
        self.mx.st = np.zeros(self.mx.num_nodes)
        for comp in self.ckt.comps:
            comp.tstep(an=self)

    def iterate(self):
        """ Run a single Newton solver """
        self.solver = TheSolverCls(self)
        try:
            v = self.solver.solve()
            print(f'New Solution {v}')
        except Exception as e:
            print(f'Failed to converge at {self.t}: {e}')
            raise e
        self.v = v
        self.history.append(self.v)

    def solve(self):
        for k in range(100):
            print(f'Solving at time {self.t}')
            self.update()
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

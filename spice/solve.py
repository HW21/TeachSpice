"""
Solver Class(es)
"""

import numpy as np
from typing import Dict, AnyStr, SupportsFloat


class MnaSystem(object):
    """
    Represents the non-linear matrix-equation
    G*x + H*g(x) = s

    And the attendant break-downs which help us solve it.
    f(x) = G*x + H*g(x) - s         # The quantity to be zero'ed
    Jf(x) = df(x)/dx = G + Jg(x)    # Definition of the Jacobian matrix `Jf(x)`
    Jf(x) * dx + f(x) = 0           # Newton update equation, to be solved for `dx`
    """

    def __init__(self, ckt, an):
        self.ckt = ckt
        self.an = an
        ckt.mx = self

        self.size = len(ckt.nodes) + len(ckt.vars)

        self.G = np.zeros((self.size, self.size))
        self.Gt = np.zeros((self.size, self.size))
        self.Jg = np.zeros((self.size, self.size))
        self.Hg = np.zeros(self.size)
        self.s = np.zeros(self.size)
        self.st = np.zeros(self.size)

    def update(self) -> None:
        """ Update non-linear component operating points """
        self.Jg = np.zeros((self.size, self.size))
        self.Hg = np.zeros(self.size)
        for comp in self.ckt.comps:
            comp.mna_update(self.an)

    def res(self, x: np.ndarray) -> np.ndarray:
        """ Return the residual error, given solution `x`. """
        return (self.G + self.Gt).dot(x) + self.Hg - self.s - self.st

    def solve(self, x: np.ndarray) -> np.ndarray:
        """ Solve our temporary-valued matrix for a change in x. """
        # print(f'G: {self.G}')
        # print(f'G-dot-x: {self.G.dot(x)}')
        # print(f'Jg: {self.Jg}')
        # print(f'Hg: {self.Hg}')
        # print(f's: {self.s}')

        lhs = self.G + self.Gt + self.Jg
        rhs = -1 * self.res(x)
        # print(f'lhs: {lhs}')
        # print(f'rhs: {rhs}')
        dx = np.linalg.solve(lhs, rhs)
        return dx


class Solver:
    """ Newton-Raphson Matrix Solver """

    def __init__(self, an, x0=None):
        self.an = an
        self.mx = an.mx
        self.x = np.array(x0, dtype='float64') if np.any(x0) else np.zeros(self.mx.size)
        self.history = [np.copy(self.x)]

    def update(self):
        """ Update non-linear component operating points """
        return self.mx.update()

    def iterate(self) -> None:
        """ Update method for Newton iterations """
        # Update non-linear component operating points
        self.update()
        # Solve Jf(x) * dx + f(x) = 0
        dx = self.mx.solve(self.x)
        # Step limiting
        MAX_STEP = 0.1
        if np.any(np.abs(dx) > MAX_STEP):
            # print(f'MAX STEPPING {np.max(np.abs(dx))}')
            dx *= MAX_STEP / np.max(np.abs(dx))
        # print(f'Updating by: {dx}')
        self.x += dx
        self.history.append(np.copy(self.x))

    def converged(self) -> bool:
        """ Convergence test, including Newton-iteration-similarity and KCL. """

        # FIXME: WTF we doing this for?
        # self.update()

        # Newton iteration similarity
        v_tol = 1e-6
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
        max_iters = 500

        for i in range(max_iters):
            # print(f'Iter #{i} - Guessing {self.x}')
            self.iterate()
            if self.converged():
                break

        if i >= max_iters - 1:
            print(self.history)
            raise Exception(f'Could Not Converge to Solution ')

        # print(f'Successfully Converged to {self.x} in {i+1} iterations')
        return self.x


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
        import scipy.optimize

        options = dict(fatol=1e-31, disp=False)
        result = scipy.optimize.minimize(fun=self.guess, x0=self.x0, method='nelder-mead', options=options)

        if not result.success:
            raise TabError(str(result))
        # print(f'Solved: {result.x}')
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

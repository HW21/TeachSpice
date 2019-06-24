import numpy as np


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

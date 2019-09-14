import numpy as np


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

        self.num_nodes = len(ckt.nodes)

        self.G = np.zeros((self.num_nodes, self.num_nodes))
        self.Gt = np.zeros((self.num_nodes, self.num_nodes))
        self.Jg = np.zeros((self.num_nodes, self.num_nodes))
        self.Hg = np.zeros(self.num_nodes)
        self.s = np.zeros(self.num_nodes)
        self.st = np.zeros(self.num_nodes)

    def update(self) -> None:
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
    """ Newton-Raphson Solver """

    def __init__(self, mx: MnaSystem, x0=None):
        self.mx = mx
        self.x = np.array(x0, dtype='float64') if np.any(x0) else np.zeros(mx.num_nodes)
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
            print(self.history)
            raise Exception(f'Could Not Converge to Solution ')

        # print(f'Successfully Converged to {self.x} in {i+1} iterations')
        return self.x

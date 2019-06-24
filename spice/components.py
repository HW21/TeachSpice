import numpy as np
from typing import Dict, AnyStr, SupportsFloat

from . import the_timestep


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

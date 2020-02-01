import numpy as np
from typing import Dict, AnyStr, SupportsFloat

from . import the_timestep


class Component(object):
    """ Base-class for (potentially non-linear) two-terminal elements. """

    linear = False
    ports = []

    def __init__(self, *, conns: dict):
        # assert list(conns.keys()) == self.ports, f'Invalid Conns {conns.keys()}, Expecting Keys {self.ports}'
        self.ckt = None
        self.conns = conns
        for name, node in conns.items():
            node.add_conn(self)

    def __repr__(self):
        return f'<{self.__class__.__name__}>'

    def get_v(self, an) -> Dict[AnyStr, SupportsFloat]:
        """ Get a dictionary of port-voltages, of the form `port_name`:`voltage`. """
        v = {}
        for name, node in self.conns.items():
            if node.solve:
                v[name] = an.v[node.num]
            else:
                v[name] = an.ckt.forces[node]
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

    def on_add(self):
        """ Call-back by `ckt` on addition of `self`."""
        pass

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

"""
i = C * dv/dt
  = C * (v[tk] - v[tk-1]) / dt
  = C * v[tk]/dt - C * v[tk-1]/dt 
"""


def mna_update_nonlinear_twoterm(comp, an):
    v_dict = comp.get_v(an)
    p = comp.conns['p']
    n = comp.conns['n']
    v = v_dict['p'] - v_dict['n']

    # Extract the type-dependent current and its derivatives
    if isinstance(comp, Diode):
        i = comp.i(v)
        di_dv = comp.di_dv(v)
    elif isinstance(comp, Capacitor):
        # Update the "resistive" part of the cap, e.g. C/dt, and
        di_dv = comp.dq_dv(v) / the_timestep

        # Update the "past current" part of the cap, e.g. V[k-1]*C/dt
        i = v * comp.c / the_timestep
        # FIXME: charge-based version
        # i = comp.q(v) / the_timestep - v * di_dv
    else:
        raise TypeError

    # Matrix Updates
    if p.solve:
        an.mx.Hg[p.num] += i
        an.mx.Jg[p.num, p.num] += di_dv
    if n.solve:
        an.mx.Hg[n.num] += i
        an.mx.Jg[n.num, n.num] += di_dv
    if p.solve and n.solve:
        an.mx.Jg[p.num, n.num] -= di_dv
        an.mx.Jg[n.num, p.num] -= di_dv


class Diode(Component):
    ports = TWO_TERM_PORTS

    def __init__(self, *, isat=1e-16, vt=25e-3, **kw):
        super().__init__(**kw)
        self.isat = isat
        self.vt = vt

    def i(self, v: Dict[AnyStr, SupportsFloat]) -> Dict[AnyStr, SupportsFloat]:
        vd = v['p'] - v['n']
        vd = self.limit(vd)
        i0 = 1 * self.isat * (np.exp(vd / self.vt) - 1)
        return dict(p=i0, n=-1 * i0)

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

    def __repr__(self):
        return f'<{self.__class__.__name__} r={self.r}>'

    def update(self, *, r: float) -> None:
        # print(f'Updating {self} to {r}')
        self.r = r

    @property
    def g(self):
        return 1 / self.r

    def i(self, v: Dict[AnyStr, SupportsFloat]) -> Dict[AnyStr, SupportsFloat]:
        vd = v['p'] - v['n']
        i0 = self.g * vd
        return dict(p=i0, n=-1 * i0)

    def mna_setup(self, an) -> None:
        p = self.conns['p']
        n = self.conns['n']
        if p.solve:
            an.mx.G[p.num, p.num] += self.g
        else:
            an.mx.s[n.num] += self.g * self.ckt.forces[p]
        if n.solve:
            an.mx.G[n.num, n.num] += self.g
        else:
            an.mx.s[p.num] += self.g * self.ckt.forces[n]
        if p.solve and n.solve:
            an.mx.G[p.num, n.num] -= self.g
            an.mx.G[n.num, p.num] -= self.g


class Capacitor(Component):
    ports = TWO_TERM_PORTS
    linear = False

    def __init__(self, *, c: float, v0: float = 0.0, **kw):
        super().__init__(**kw)
        self.c = c
        self.v0 = v0
        self.vm1 = v0

    def q(self, v: float) -> float:
        return self.c * v

    def dq_dv(self, v: float) -> float:
        return self.c

    def i(self, v: Dict[AnyStr, SupportsFloat]) -> Dict[AnyStr, SupportsFloat]:
        vd = v['p'] - v['n']
        i0 = self.c * (vd - self.vm1) / the_timestep
        return dict(p=i0, n=-1 * i0)

    def tstep(self, an) -> None:
        p = self.conns['p']
        n = self.conns['n']
        v = self.get_v(an)
        vd = v['p'] - v['n']
        self.vm1 = vd
        # Update the "past charge" part of the cap equation
        rhs = - 1 * self.q(vd) / the_timestep
        if p.solve:
            an.mx.st[p.num] = -1 * rhs
        if n.solve:
            an.mx.st[n.num] = 1 * rhs

    def mna_update(self, an) -> None:
        return mna_update_nonlinear_twoterm(self, an)


class Vsrc(Component):
    """ Constant Voltage Source """

    ports = TWO_TERM_PORTS
    linear = True

    def __init__(self, *, vdc: float, **kw):
        super().__init__(**kw)
        self.vdc = vdc
        self.ivar = None

    def i(self, v: Dict[AnyStr, SupportsFloat]) -> Dict[AnyStr, SupportsFloat]:
        return dict()

    def on_add(self):
        self.ivar = self.ckt.add_variable()

    def mna_setup(self, an):
        mx = an.mx
        p = self.conns['p']
        n = self.conns['n']
        iindex = self.ivar + len(self.ckt.nodes)

        mx.s[iindex] = self.vdc
        if p.solve:
            mx.G[p.num, iindex] += 1.0
            mx.G[iindex, p.num] += 1.0
        if n.solve:
            mx.G[n.num, iindex] -= 1.0
            mx.G[iindex, n.num] -= 1.0


class Isrc(Component):
    """ Constant Current Source """

    ports = TWO_TERM_PORTS
    linear = True

    def __init__(self, *, idc: float, **kw):
        super().__init__(**kw)
        self.idc = idc

    def i(self, v: Dict[AnyStr, SupportsFloat]) -> Dict[AnyStr, SupportsFloat]:
        return dict(p=-1 * self.idc, n=self.idc)

    def mna_setup(self, an):
        s = an.mx.s
        p = self.conns['p']
        n = self.conns['n']
        assert not n.solve  # No floating sources, yet
        s[p.num] = 1 * self.idc


# class Bjt(Component):
#     ports = ['c', 'b', 'e']


class Mos(Component):
    """ Level-Zero MOS Model """

    ports = ['g', 'd', 's', 'b']
    vth = 0.25
    beta = 50e-3
    lam = 1.0 / 30  # Sorry "lambda" is a Python language keyword

    def __init__(self, *, polarity=1, **kwargs):
        super().__init__(**kwargs)
        self.polarity = 1 if polarity > 0 else -1

    def i(self, v: Dict[AnyStr, SupportsFloat]) -> Dict[AnyStr, SupportsFloat]:
        """ Calculate operating-point dict from voltage-dict """

        vds = self.polarity * (v['d'] - v['s'])
        vds = min(vds, 1.0)
        vgs = self.polarity * (v['g'] - v['s'])
        vgs = min(vgs, 1.0)
        vov = vgs - self.vth

        reversed = bool(vds < 0)
        if reversed: vds = -1 * vds
        rev = -1 if reversed else +1

        if vov <= 0:  # Cutoff
            mode = 'CUTOFF'
            ids = 0
            gm = 0
            gds = 0
        elif vds >= vov:  # Saturation
            mode = 'SAT'
            ids = self.beta / 2 * (vov ** 2) * (1 + self.lam * vds)
            gm = self.beta * vov * (1 + self.lam * vds)
            gds = self.lam * self.beta / 2 * (vov ** 2)
        else:  # Triode
            mode = 'TRIODE'
            ids = self.beta * ((vov * vds) - (vds ** 2) / 2) * (1 + self.lam * vds)
            gm = self.beta * vds * (1 + self.lam * vds)
            gds = self.beta * ((vov - vds) * (1 + self.lam * vds) + self.lam * ((vov * vds) - (vds ** 2) / 2))

        rds = np.NaN if gds == 0 else 1 / gds
        ids = rev * self.polarity * ids
        d_ = {"d": ids, "s": -1 * ids, "g": 0.0, "b": 0.0}
        # print(f'Op Point: {d_}')
        return d_

    def op_point(self, v: dict) -> dict:
        """ Calculate operating-point dict from voltage-dict """

        vds = self.polarity * (v['d'] - v['s'])
        vds = min(vds, 1.0)
        vgs = self.polarity * (v['g'] - v['s'])
        vgs = min(vgs, 1.0)
        vov = vgs - self.vth

        reversed = bool(vds < 0)
        if reversed: vds = -1 * vds

        if vov <= 0:  # Cutoff
            mode = 'CUTOFF'
            ids = 0
            gm = 0
            gds = 0
        elif vds >= vov:  # Saturation
            mode = 'SAT'
            ids = self.beta / 2 * (vov ** 2) * (1 + self.lam * vds)
            gm = self.beta * vov * (1 + self.lam * vds)
            gds = self.lam * self.beta / 2 * (vov ** 2)
        else:  # Triode
            mode = 'TRIODE'
            ids = self.beta * ((vov * vds) - (vds ** 2) / 2) * (1 + self.lam * vds)
            gm = self.beta * vds * (1 + self.lam * vds)
            gds = self.beta * ((vov - vds) * (1 + self.lam * vds) + self.lam * ((vov * vds) - (vds ** 2) / 2))

        rds = np.NaN if gds == 0 else 1 / gds
        d_ = {"ids": ids, "gds": gds, "gm": gm, "rds": rds, "mode": mode, 'rev': reversed}
        # print(f'Op Point: {d_}')
        return d_

    def mna_update(self, an) -> None:
        v = self.get_v(an)
        op = self.op_point(v)
        ids = op['ids']
        gds = op['gds']
        gm = op['gm']
        rev = -1 if op['rev'] else +1

        v = self.get_v(an)  # FIXME
        mx = an.mx

        d = self.conns['d']
        s = self.conns['s']
        g = self.conns['g']
        b = self.conns['b']
        assert not b.solve  # No floating bulk, yet

        # FIXME: all these signs are suspect
        if d.solve:
            mx.Hg[d.num] += rev * self.polarity * ids
            mx.Jg[d.num, d.num] += gds  ##self.polarity * gds
        if s.solve:
            mx.Hg[s.num] -= rev * self.polarity * ids
            mx.Jg[s.num, s.num] -= rev * self.polarity * (gm + gds)
        # else:
        #     mx.Hg[d.num] += gds * v['s'] ##'-= self.polarity * gds * v['s']
        if d.solve and s.solve:
            mx.Jg[d.num, s.num] += rev * self.polarity * (gm + gds)
            mx.Jg[s.num, d.num] += rev * self.polarity * gds
        if g.solve and s.solve:
            mx.Jg[s.num, g.num] -= rev * self.polarity * gm
        if g.solve and d.solve:
            mx.Jg[d.num, g.num] += rev * self.polarity * gm

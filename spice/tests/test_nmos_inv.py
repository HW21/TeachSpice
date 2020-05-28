import numpy as np

from .. import DcOp


def nmos_inv(vgs: float):
    """ Create an instance of an NMOS-resistor inverter, with Vgs=`vgs`. """
    from .. import Circuit, Resistor, Mos, Vsrc

    class NmosInv(Circuit):
        """ NMOS-Resistor Inverter """

        def define(self):
            [g, d, vdd] = self.create_nodes(3)

            self.create_comp(cls=Vsrc, vdc=1.0, conns=dict(p=vdd, n=self.node0))
            self.create_comp(cls=Vsrc, vdc=vgs, conns=dict(p=g, n=self.node0))
            self.create_comp(cls=Mos, polarity=1,
                             conns={'s': self.node0, 'b': self.node0,
                                    'd': d, 'g': g})
            self.create_comp(cls=Resistor, r=10e3,
                             conns=dict(p=vdd, n=d))

    return NmosInv()


def test_dcop():
    dut = nmos_inv(0.0)
    s = DcOp(ckt=dut)

    y = s.solve()
    assert np.allclose(y, [0, 1.0, 1.0, 0, 0])  # g, d, vdd, two Vsrc currents


def test_dc_sweep():
    vg = []
    vd = []
    vds = 0.0  # 1.0
    for k in range(11):
        vgs = k / 10.0
        dut = nmos_inv(vgs)
        s = DcOp(ckt=dut)

        s.solve([vds])
        vds = s.v[1]
        vg += [vgs]
        vd += [vds]

    assert (vd[0] > 0.9)
    assert (vd[-1] < 0.1)

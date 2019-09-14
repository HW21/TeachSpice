from .. import Circuit, DcOp, Resistor, Mos
from ..analysis import Contour


def nmos_inv(vgs):
    class NmosInv(Circuit):
        """ NMOS-Resistor Inverter """

        def define(self):
            self.create_nodes(1)
            vdd = self.create_forced_node(name='vdd', v=1.0)
            g = self.create_forced_node(name='vgs', v=vgs)

            self.create_comp(cls=Mos, polarity=1,
                             conns={'s': self.node0, 'b': self.node0,
                                    'd': self.nodes[0], 'g': g})
            self.create_comp(cls=Resistor, r=10e3,
                             conns=dict(p=vdd, n=self.nodes[0], ), )

    return NmosInv()


def test_nmos_inv():
    vg = []
    vd = []
    vds = 1.0
    for k in range(11):
        vgs = k / 10.0
        dut = nmos_inv(vgs)
        s = DcOp(ckt=dut)

        s.solve([vds])
        vds = s.v[0]
        vg += [vgs]
        vd += [vds]
    print(vg)
    print(vd)

    assert (vd[0] > 0.9)
    assert (vd[-1] < 0.1)


# def test_nmos_inv_contour():
#     xs = []
#     dxs = []
#     for k in range(11):
#         vgs = k / 10.0
#         dut = nmos_inv(vgs)
#         an = Contour(dut)
#         x, dx = an.explore()
#         xs.append(x)
#         dxs.append(dx)
#     print(xs)
#     print(dxs)

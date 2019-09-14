import numpy as np
from .. import Circuit, DcOp, Resistor, Mos
from ..analysis import Contour


def cmos_inv(vgs):
    class CmosInv(Circuit):
        """ Cmos Inverter """

        def define(self):
            self.create_nodes(1)
            vdd = self.create_forced_node(name='vdd', v=1.0)
            g = self.create_forced_node(name='vgs', v=vgs)

            self.create_comp(cls=Mos, polarity=-1,
                             conns={'s': vdd, 'b': vdd,
                                    'd': self.nodes[0], 'g': g})
            self.create_comp(cls=Mos, polarity=1,
                             conns={'s': self.node0, 'b': self.node0,
                                    'd': self.nodes[0], 'g': g})

    return CmosInv()


def test_cmos_inv():
    vis = []
    vos = []
    vo = [1.0]
    for vi in np.linspace(0, 1.0, 101):
        dut = cmos_inv(vi)
        s = DcOp(ckt=dut)

        s.solve([vo])
        vo = s.v[0]
        vis += [vi]
        vos += [vo]
    print(vis)
    print(vos)

    assert (vos[0] > 0.9)
    assert (vos[-1] < 0.1)

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

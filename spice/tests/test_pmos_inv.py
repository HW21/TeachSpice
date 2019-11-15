from .. import Circuit, DcOp, Resistor, Mos, Diode
from ..analysis import Contour


def pmos_inv(vgs):
    class PmosInv(Circuit):
        """ PMOS-Resistor Inverter """

        def define(self):
            self.create_nodes(1)
            vdd = self.create_forced_node(name='vdd', v=1.0)
            g = self.create_forced_node(name='vgs', v=vgs)

            self.create_comp(cls=Mos, polarity=-1,
                             conns={'s': vdd, 'b': vdd,
                                    'd': self.nodes[0], 'g': g})
            self.create_comp(cls=Resistor, r=10e3,
                             conns=dict(p=self.node0, n=self.nodes[0], ), )
            # self.create_comp(cls=Diode,
            #                  conns=dict(p=self.node0, n=self.nodes[0], ), )
            # self.create_comp(cls=Diode,
            #                  conns=dict(p=self.nodes[0], n=vdd, ), )

    return PmosInv()


def test_dcop():
    vds = 0.0  # 1.0
    vgs = 0.0
    dut = pmos_inv(vgs)
    s = DcOp(ckt=dut)

    y = s.solve([vds])
    print(y)


def test_pmos_inv():
    vg = []
    vd = []
    vds = 0.0
    for k in range(11):
        vgs = k / 10.0
        print(f'Running vgs={vgs}')
        dut = pmos_inv(vgs)
        s = DcOp(ckt=dut)

        s.solve()
        # s.solve([vds])

        vds = s.v[0]
        vg += [vgs]
        vd += [vds]
    print(vg)
    print(vd)

    assert (vd[0] > 0.9)
    assert (vd[-1] < 0.1)


# def test_pmos_inv_contour():
#     xs = []
#     ys = []
#     dxs = []
#     for k in range(11):
#         vgs = k / 10.0
#         dut = pmos_inv(vgs)
#         an = Contour(dut)
#         x, y, dx = an.explore()
#         xs.append(x)
#         dxs.append(dx)
#     print(xs)
#     print(ys)
#     print(dxs)

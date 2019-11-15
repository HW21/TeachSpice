from .. import Circuit, DcOp, Resistor, Mos


def nmos_latch():
    class NmosLatch(Circuit):
        """ NMOS-Resistor Latch """

        def define(self):
            self.create_nodes(2)
            vdd = self.create_forced_node(name='vdd', v=1.0)

            self.create_comp(cls=Mos, polarity=1,
                             conns={'s': self.node0, 'b': self.node0,
                                    'd': self.nodes[0], 'g': self.nodes[1]})
            self.create_comp(cls=Resistor, r=10e3,
                             conns=dict(p=vdd, n=self.nodes[0], ), )
            self.create_comp(cls=Mos, polarity=1,
                             conns={'s': self.node0, 'b': self.node0,
                                    'd': self.nodes[1], 'g': self.nodes[0]})
            self.create_comp(cls=Resistor, r=10e3,
                             conns=dict(p=vdd, n=self.nodes[1], ), )

    return NmosLatch()


def test_dcop():
    dut = nmos_latch()
    s = DcOp(ckt=dut)

    s.solve([0,0.15])
    print(s.v)


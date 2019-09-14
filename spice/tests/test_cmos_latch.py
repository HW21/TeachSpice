import numpy as np
from .. import Circuit, Node, DcOp, Resistor, Capacitor, Diode, Isrc, Mos


class CmosLatch(Circuit):
    """ Back-to-Back CMOS Inverter Latch """

    def define(self):
        self.create_nodes(2)
        vdd = self.create_forced_node(name='vdd', v=1.0)

        self.create_comp(cls=Mos, polarity=1,
                         conns={'s': self.node0, 'b': self.node0,
                                'd': self.nodes[1], 'g': self.nodes[0]})
        self.create_comp(cls=Mos, polarity=-1,
                         conns={'s': vdd, 'b': vdd,
                                'd': self.nodes[1], 'g': self.nodes[0]})

        self.create_comp(cls=Mos, polarity=1,
                         conns={'s': self.node0, 'b': self.node0,
                                'd': self.nodes[0], 'g': self.nodes[1]})
        self.create_comp(cls=Mos, polarity=-1,
                         conns={'s': vdd, 'b': vdd,
                                'd': self.nodes[0], 'g': self.nodes[1]})


def sim(x0=None):
    """ Create an instance of the circuit under test, and simulate it. """
    ckt = CmosLatch()
    s = DcOp(ckt=ckt)
    s.solve(x0)
    print(s.v)
    return s


def test_stuff():
    """ Run some sims!
    Sweep and plot a few RC parameters. """
    sim([0.5, 0.5])

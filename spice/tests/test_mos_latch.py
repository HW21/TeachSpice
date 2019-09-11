import numpy as np
from .. import Circuit, Node, DcOp, Resistor, Capacitor, Diode, Isrc, Mos




class MosLatch(Circuit):
    """ Back-to-Back Inverter Latch """

    def define(self):
        self.create_nodes(2)
        vdd = self.create_forced_node(name='vdd', v=1.0)

        self.create_comp(cls=Mos, polarity=1,
                         conns={'s': self.node0, 'b': self.node0,
                                'd': self.nodes[1], 'g': self.nodes[0]})
        # self.create_comp(cls=Mos, polarity=-1,
        #                  conns={'s': vdd, 'b': vdd,
        #                         'd': self.nodes[1], 'g': self.nodes[2]})
        self.create_comp(cls=Resistor, r=1e3,
                         conns=dict(
                             p=vdd,
                             n=self.nodes[1],
                         ), )

        self.create_comp(cls=Mos, polarity=1,
                         conns={'s': self.node0, 'b': self.node0,
                                'd': self.nodes[0], 'g': self.nodes[1]})
        # self.create_comp(cls=Mos, polarity=-1,
        #                  conns={'s': vdd, 'b': vdd,
        #                         'd': self.nodes[2], 'g': self.nodes[1]})
        self.create_comp(cls=Resistor, r=1e3,
                         conns=dict(
                             p=vdd,
                             n=self.nodes[0],
                         ), )


def sim(x0=None):
    """ Create an instance of the circuit under test, and simulate it. """
    ckt = MosLatch()
    s = DcOp(ckt=ckt)
    s.solve(x0)
    return s


def test_stuff():
    """ Run some sims!
    Sweep and plot a few RC parameters. """
    sim([0, 0.1])

    # import pandas as pd
    # import matplotlib.pyplot as plt
    #
    # def add_tran_plots(sim):
    #     df = pd.DataFrame.from_records(sim.history)
    #     # print(df)
    #     plt.plot(df[0])
    #     plt.plot(df[1])
    #
    # for c in 1e-14, 1e-13, 1e-12:
    #     s = sim(c=c)
    #     add_tran_plots(s)
    #
    # # plt.show()

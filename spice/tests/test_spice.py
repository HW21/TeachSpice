from .. import Circuit, Node, Tran, Resistor, Capacitor, Diode, Isrc


def rc_filter(c=1e-12, num_nodes=1):
    class RcFilter(Circuit):
        """ Current-Driven RC Filter,
        with Parameterizable R, C, and # intermediate nodes """

        def define(self):
            self.create_nodes(num_nodes)

            for k in range(num_nodes - 1):
                self.create_comp(cls=Resistor,
                                 r=(num_nodes - 1) * 1e3,
                                 conns=dict(
                                     p=self.nodes[k + 1],
                                     n=self.nodes[k],
                                 ), )

            self.create_comp(cls=Resistor,
                             r=1e3,
                             conns=dict(
                                 p=self.nodes[0],
                                 n=self.node0,
                             ), )
            self.create_comp(cls=Isrc,
                             idc=1e-3,
                             conns=dict(
                                 p=self.nodes[num_nodes - 1],
                                 n=self.node0,
                             ), )

            self.create_comp(cls=Capacitor,
                             c=c,
                             conns=dict(
                                 p=self.nodes[0],
                                 n=self.node0,
                             ), )

    return RcFilter()


def sim(c=1e-12):
    """ Create an instance of the circuit under test, and simulate it. """
    num_nodes = 2
    ckt = rc_filter(c=c, num_nodes=num_nodes)

    s = Tran(ckt=ckt)
    s.solve()

    return s


def test_stuff():
    """ Run some sims!
    Sweep and plot a few RC parameters. """
    import pandas as pd
    import matplotlib.pyplot as plt

    def add_tran_plots(sim):
        df = pd.DataFrame.from_records(sim.history)
        # print(df)
        plt.plot(df[0])
        plt.plot(df[1])

    for c in 1e-14, 1e-13, 1e-12:
        s = sim(c=c)
        add_tran_plots(s)

    # plt.show()

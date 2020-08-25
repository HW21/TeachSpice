import numpy as np
from .. import Circuit, Vsrc, Resistor, DcOp


def test_vsrc1():
    """ Test the DcOp solution of a Circuit with *just* an independent Vsrc. """
    c = Circuit()
    c.create_nodes(1)
    c.create_comp(cls=Vsrc, vdc=1.1,
                  conns={'p': c.nodes[0], 'n': c.node0})
    an = DcOp(ckt=c)
    x = an.solve()
    assert len(x) == 2
    assert np.allclose(x, [1.1, 0.0])


def test_vsrc2():
    """ Vsrc + Resistor """
    c = Circuit()
    c.create_nodes(1)
    c.create_comp(cls=Vsrc, vdc=1.1,
                  conns={'p': c.nodes[0], 'n': c.node0})
    c.create_comp(cls=Resistor, r=1e3,
                  conns={'p': c.nodes[0], 'n': c.node0})
    an = DcOp(ckt=c)
    x = an.solve()
    assert len(x) == 2
    assert np.allclose(x, [1.1, -1.1e-3])


def test_vsrc3():
    """ Three sources in series """
    c = Circuit()
    c.create_nodes(3)
    c.create_comp(cls=Vsrc, vdc=11.1,
                  conns={'p': c.nodes[0], 'n': c.node0})
    c.create_comp(cls=Vsrc, vdc=3.14,
                  conns={'p': c.nodes[1], 'n': c.nodes[0]})
    c.create_comp(cls=Vsrc, vdc=0.375,
                  conns={'p': c.nodes[2], 'n': c.nodes[1]})
    an = DcOp(ckt=c)
    x = an.solve()
    assert len(x) == 6
    assert np.allclose(x, [11.1, 11.1 + 3.14, 11.1 + 3.14 + 0.375, 0, 0, 0])

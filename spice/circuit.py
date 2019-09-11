import numpy as np
from typing import Dict, AnyStr, SupportsFloat

from .components import Component, Capacitor, Diode

from . import the_timestep


class Node(object):
    """ Circuit Node """

    def __init__(self):
        self.conns = []
        self.num = None
        self.ckt = None
        self.solve = True

    def add_conn(self, conn: Component):
        assert isinstance(conn, Component)
        self.conns.append(conn)

    def limit(self, v: float) -> float:
        for c in self.conns:
            v = c.limit(v)
        return v


class Circuit(object):
    """ Circuit Under Test for Simulation. """

    def __init__(self):
        self.comps = []
        self.nodes = []
        self.forces = {}
        # Create node zero
        self.node0 = self.create_forced_node(name='gnd', v=0.0)
        # Run any custom definition
        self.define()

    def define(self) -> None:
        """ Sub-classes can add content here. """
        pass

    def create_nodes(self, num: int):
        for k in range(num):
            self.add_node(Node())

    def add_node(self, node: Node) -> int:
        assert isinstance(node, Node)
        node.num = len(self.nodes)
        node.ckt = self
        self.nodes.append(node)
        return len(self.nodes) - 1

    def create_forced_node(self, name: str, v: float) -> Node:
        """ Create and return a forced node, at voltage `v` """
        node = Node()
        node.num = len(self.forces)
        node.name = name
        node.solve = False
        node.ckt = self
        self.forces[node] = v
        return node

    def create_comp(self, *, cls: type, **kw) -> Component:
        """ Create and add Component of class `cls`, and return it. """
        comp = cls(**kw)
        comp.ckt = self
        return self.add_comp(comp=comp)

    def add_comp(self, comp: Component) -> Component:
        """ Add Component `comp`, and return it. """
        assert isinstance(comp, Component)
        for name, node in comp.conns.items():
            assert node in self.nodes or node in self.forces
        self.comps.append(comp)
        return comp

"""
Support for storing array-of-entries form matrices to YAML
"""

from pathlib import Path
from typing import List, Tuple, Optional

import ruamel.yaml

yaml = ruamel.yaml.YAML()


@yaml.register_class
class MatrixYaml(object):
    def __init__(self):
        self.desc: str = ""
        self.size: int = 0
        self.entries: List[Tuple] = []
        self.rhs: Optional[List[float]] = None
        self.ntype: str = "real"
        self.solution: Optional[List[float]] = None

    @classmethod
    def from_sparse_file(cls, sf):
        from .file import SparseFile
        if not isinstance(sf, SparseFile):
            raise TypeError(sf)

        self = MatrixYaml()
        self.desc = sf.desc
        self.size = sf.size
        self.entries = [(r - 1, c - 1, v) for (r, c, v) in sf.entries]
        self.rhs = sf.rhs
        self.ntype = sf.ntype
        return self

    def to_dict(self):
        return dict(
            desc=self.desc,
            size=self.size,
            ntype=self.ntype,
            entries=self.entries,
            rhs=self.rhs,
            solution=self.solution,
        )

    @classmethod
    def to_yaml(cls, representer, node):
        d = node.to_dict()
        return representer.represent(d)

    @classmethod
    def from_dict(cls, d: dict):
        self = cls()
        self.desc = d['desc']
        self.size = d['size']
        self.entries = d['entries']
        self.rhs = d['rhs']
        self.ntype = d['ntype']
        self.solution = d['solution']
        return self

    def to_mat(self):
        from .matrix import SparseMatrix
        m = SparseMatrix()
        for e in self.entries:
            m.add_element(*e)
        assert len(m.rows) == self.size
        assert len(m.cols) == self.size
        return m

    def dump(self, file):
        p = Path(file)
        yaml.dump([self], p)

    @classmethod
    def load(cls, file):
        p = Path(file)
        y = yaml.load(p)
        rhs = None
        if y['rhs']:
            rhs = list(y['rhs'])
        solution = None
        if y['solution']:
            solution = list(y['solution'])
        d = dict(
            desc=str(y['desc']),
            ntype=str(y['ntype']),
            size=int(y['size']),
            entries=list(y['entries']),
            rhs=rhs,
            solution=solution,
        )
        return cls.from_dict(d)

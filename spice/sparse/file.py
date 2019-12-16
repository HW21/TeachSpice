from typing import List, Tuple, Optional


class SparseFile(object):
    def __init__(self, path):
        self.path = path
        self.desc: str = ""
        self.size: int = 0
        self.entries: List[Tuple] = []
        self.rhs: Optional[List[float]] = None
        self.ntype: str = "real"
        self.line: int = 1

    def read(self):
        with open(self.path) as f:

            self.desc = f.readline().strip()
            self.line = 2
            line2 = f.readline()
            line2_entries = line2.strip().split()
            if len(line2_entries) == 2:
                self.size = int(line2_entries[0])
                if line2_entries[1] != "real":
                    raise NotImplementedError("only real values supported, for now")
            elif "--" in line2_entries:
                self.desc2 = line2
                self.line = 3
                self.size = int(f.readline().strip())
            else:
                raise Exception(f"Header parse error: {line2}")

            def read_entry():
                self.line += 1
                line = f.readline()
                row, col, val = line.strip().split()
                row = int(row)
                col = int(col)
                val = float(val)
                return (row, col, val)

            entry = read_entry()
            while entry[0] != 0 and entry[1] != 0:
                self.entries.append(entry)
                entry = read_entry()

            self.line += 1
            line = f.readline().strip()
            if line:  # Read a RHS
                self.rhs = []
                while line:
                    if "Beginning source vector" not in line:
                        self.rhs.append(float(line))
                    self.line += 1
                    line = f.readline().strip()
                assert len(self.rhs) == self.size, "Matrix and RHS size do not match"

    def to_mat(self):
        from .matrix import SparseMatrix
        m = SparseMatrix()
        for r, c, v in self.entries:
            m.add_element(r - 1, c - 1, v)
        return m

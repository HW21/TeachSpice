from typing import Dict, List, Tuple, Union, Optional
from enum import IntEnum, auto


class Element(object):
    def __init__(self, row: int, col: int, val: float, fillin: bool = False):
        self.row = row
        self.col = col
        self.val = val
        self.fillin = fillin
        self.next_in_row = None
        self.next_in_col = None

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col and self.val == other.val

    def __repr__(self):
        return f"<{self.__class__.__name__}(row={self.row}, col={self.col}, val={self.val}, id={id(self)})>"


class MatrixState(IntEnum):
    CREATED = auto()
    FACTORING = auto()
    FACTORED = auto()


class SparseMatrix(object):
    def __init__(self):
        self.state = MatrixState.CREATED
        self.rows: List[Optional[Element]] = [None]
        self.cols: List[Optional[Element]] = [None]
        self.diag: List[Optional[Element]] = [None]
        self.fillins: List[Element] = []
        self.rowmap_i2e: Optional[List[int]] = None
        self.rowmap_e2i: Optional[List[int]] = None
        self.row_swap_history: List[Tuple[float, float]] = []

    def elements(self):
        """ Columns-first iterator of elements """
        for c in self.cols:
            while c is not None:
                yield c
                c = c.next_in_col

    def __eq__(self, other):
        if len(self.rows) != len(other.rows): return False
        if len(self.cols) != len(other.cols): return False
        # FIXME: means to check these lengths, without pulling all into memory
        s = list(self.elements())
        o = list(other.elements())
        if len(s) != len(o): return False
        for se, oe in zip(s, o):
            if se != oe: return False
        return True

    def get(self, row: int, col: int) -> Optional[Element]:
        """ Get the element at (row,col), or None if no element present """
        if row < 0: return None
        if col < 0: return None
        if row > len(self.rows) - 1: return None
        if col > len(self.cols) - 1: return None

        # Easy access cases
        # if row == col: return self.diag[row] # FIXME

        # Real search
        e = self.rows[row]
        while e is not None and e.col < col:
            e = e.next_in_row
        if e is None or e.col != col:
            return None
        assert e.row == row
        assert e.col == col
        return e

    def swap_rows(self, x: int, y: int):
        if x == y: return
        if y < x: x, y = y, x

        ex = self.rows[x]
        ey = self.rows[y]

        while ex is not None or ey is not None:
            if ex is None:
                self.move_row(ey, x)
                ey = ey.next_in_row
            elif ey is None or ex.col < ey.col:
                self.move_row(ex, y)
                ex = ex.next_in_row
            elif ey.col < ex.col:
                self.move_row(ey, x)
                ey = ey.next_in_row
            else:
                self.exchange_col_elements(ex, ey)
                ex = ex.next_in_row
                ey = ey.next_in_row

        # Swap row-header pointers
        self.rows[x], self.rows[y] = self.rows[y], self.rows[x]

        # Make updates to our row-mappings
        self.row_swap_history.append((x, y))
        self.rowmap_i2e[x], self.rowmap_i2e[y] = self.rowmap_i2e[y], self.rowmap_i2e[x]
        self.rowmap_e2i[self.rowmap_i2e[x]] = x
        self.rowmap_e2i[self.rowmap_i2e[y]] = y

    def above(self, e: Element, hint: Optional[Element] = None) -> Optional[Element]:
        """ Find the element above `e`.
        Optional hint provides an element that is as close as we know to right.
        If e is the first element in its column, or hint is e, returns None. """
        prev = hint or self.cols[e.col]
        if prev is None or prev is e: return None

        next = prev.next_in_col
        while next is not None and next is not e:
            prev = next
            next = next.next_in_col
        assert next is e
        return prev

    def above_row(self, row: int, col: int, hint: Optional[Element] = None) -> Optional[Element]:
        prev = hint or self.cols[col]
        if prev is None or prev.row >= row: return None

        next = prev.next_in_col
        while next is not None and next.row < row:
            prev = next
            next = next.next_in_col
        return prev

    def move_row(self, e: Element, row: int):
        """ Move element `e` to row `row`, updating pointers accordingly.
        There must not be an existing element at (row, e.col).
        However there may or may not be entries above or below `row` and/or `e`. """
        if e.row == row: return

        if e.row < row:  # Search for element first
            be = self.above(e)
            br = self.above_row(row, col=e.col, hint=e)
        else:  # Search for row first
            br = self.above_row(row, col=e.col)
            be = self.above(e, hint=br)

        # Things that (may) need to happen:
        # * "Short" over `e` from `be` to `e.next`
        # * Splice `e` in after `br`, or
        # * Update column-header

        if br is not be:  # If we (may) need some pointer updates
            if be is not None:  # Short-circuit over `e`
                be.next_in_col = e.next_in_col

            if br is None:  # New first in column
                e.next_in_col = self.cols[e.col]
                self.cols[e.col] = e
            elif br is not e:  # Splice `e` in after `br`
                e.next_in_col = br.next_in_col
                br.next_in_col = e

        e.row = row
        if e.row == e.col: self.diag[e.row] = e

    def exchange_col_elements(self, ex: Element, ey: Element):
        """ Swap the rows of two elements in the same column """
        assert ex.col == ey.col
        assert ex.row < ey.row

        # Find the elements before each of `ex` and `ey`.
        bx = self.above(ex)
        by = self.above(ey, hint=ex)

        # Now we can get to swappin.
        ex.row, ey.row = ey.row, ex.row

        if bx is None:  # If `ex` is the *first* entry in the column, replace it to our header-list
            self.cols[ex.col] = ey
        else:  # Otherwise patch ey into bx
            bx.next_in_col = ey

        if by is ex:  # `ex` and `ey` are adjacent
            ey.next_in_col, ex.next_in_col = ex, ey.next_in_col
        else:  # Elements in-between `ex` and `ey`.  Update the last one.
            ey.next_in_col, ex.next_in_col = ex.next_in_col, ey.next_in_col
            by.next_in_col = ex

        if ex.row == ex.col: self.diag[ex.row] = ex
        if ey.row == ey.col: self.diag[ey.row] = ey

    def insert(self, e: Element) -> Element:
        expanded = False
        if e.row > len(self.rows) - 1:
            self.rows.extend([None] * (1 + e.row - len(self.rows)))
            expanded = True
        if e.col > len(self.cols) - 1:
            self.cols.extend([None] * (1 + e.col - len(self.cols)))
            expanded = True
        if expanded:
            new_diag_len = min(len(self.rows), len(self.cols))
            self.diag.extend([None] * (new_diag_len - len(self.diag)))
        if e.row == e.col:
            self.diag[e.col] = e
        if e.fillin:
            self.fillins.append(e)

        # Insert into the col
        col_head = self.cols[e.col]
        if col_head is None:
            self.cols[e.col] = e
        elif col_head.col > e.col:
            e.next_in_col = col_head
            self.cols[e.col] = e
        else:
            elem = col_head
            next = col_head.next_in_col
            while next is not None and next.row < e.row:
                elem = next
                next = next.next_in_col
            # Now elem and next straddle e.col
            elem.next_in_col = e
            e.next_in_col = next

        # Insert into the row
        row_head = self.rows[e.row]
        if row_head is None:
            self.rows[e.row] = e
        elif row_head.col > e.col:
            e.next_in_row = row_head
            self.rows[e.row] = e
        else:
            elem = row_head
            next = row_head.next_in_row
            while next is not None and next.col < e.col:
                elem = next
                next = next.next_in_row
            # Now elem and next straddle e.col
            elem.next_in_row = e
            e.next_in_row = next

        return e

    def lu_factorize(self):
        """ Updates self to S = L + U - I.
        Diagonal entries are those of U;
        L has diagonal entries equal to one. """
        # FIXME: no pivoting/ swapping, yet

        # Quick singularity check
        # Each row & column must have entries
        for e in self.cols:
            if e is None: raise SingularMatrix
        for e in self.rows:
            if e is None: raise SingularMatrix

        self.state = MatrixState.FACTORING

        # Set up row-swap mappings
        self.rowmap_e2i = list(range(len(self.rows)))
        self.rowmap_i2e = list(range(len(self.rows)))

        for d in self.diag[:-1]:
            pivot = self.find_max_below(d)
            self.swap_rows(pivot.row, d.row)
            self.row_col_elim(pivot)
        self.state = MatrixState.FACTORED

    def find_max_below(self, below: Element):
        """ Find the max in-column value at or below Element `e` """
        e = max_elem = below
        while e is not None:
            if abs(e.val) > abs(max_elem.val):
                max_elem = e
            e = e.next_in_col
        return max_elem

    def row_col_elim(self, pivot: Element):
        """ Eliminate the row & column of element `pivot`,
        transforming them into the LU-factored row/col of L and U.
        Uses Gauss's algorithm, without any fancy tricks applied. """
        if pivot.val == 0: raise SingularMatrix

        # Divide the pivot-column entries by the pivot-value
        plower = pivot.next_in_col
        while plower is not None:
            plower.val /= pivot.val
            plower = plower.next_in_col

        pupper = pivot.next_in_row
        while pupper is not None:
            plower = pivot.next_in_col
            # pabove = pupper
            psub = pupper.next_in_col
            while plower is not None:

                while psub is not None and psub.row < plower.row:
                    # pabove = psub
                    psub = psub.next_in_col
                if psub is None or psub.row > plower.row:
                    psub = self.add_element(row=plower.row, col=pupper.col, val=0.0, fillin=True)

                psub.val -= pupper.val * plower.val
                psub = psub.next_in_col
                plower = plower.next_in_col
            pupper = pupper.next_in_row

    def solve(self, rhs: Union[Dict[int, float], List[float], None] = None):
        """ Complete forward & backward substitution
        Generally described as breaking Ax = LUx = b down into
        Lc = b, Ux = c """
        assert self.state == MatrixState.FACTORED

        c = [0.0] * len(self.rows)
        if isinstance(rhs, list):
            assert len(rhs) == len(self.rows), f'Invalid rhs: length {len(rhs)} for matrix size {len(self.rows)}'
            c = rhs
            for n, v in enumerate(rhs):
                c[self.rowmap_e2i[n]] = v
        else:  # Collect a sparse rhs into a solution-list
            if rhs is not None:
                for r, v in rhs.items():
                    c[self.rowmap_e2i[r]] = v

        # Forward substitution: Lc=b
        for d in self.diag:
            # Walk down each column, update c
            if c[d.row] == 0:  # No updates to make on this iteration
                continue
            # c[d.row] /= d.val
            e = d.next_in_col
            while e is not None:
                c[e.row] -= c[d.row] * e.val
                e = e.next_in_col

        # Backward substitution: Ux=c
        for d in self.diag[::-1]:
            # Walk each row, update c
            e = d.next_in_row
            while e is not None:
                c[e.row] -= c[e.col] * e.val
                e = e.next_in_row
            c[d.row] /= d.val

        return c

    def matmul(self, other):
        """ Matrix multiplication self*other """
        if len(self.rows) != len(other.cols):
            raise MatrixDimError

        m = SparseMatrix()
        for (row, re) in enumerate(self.rows):
            for (col, ce) in enumerate(other.cols):
                val = 0

                # "Two pointer" row * col dot-product
                while re is not None and ce is not None:
                    if ce is None or re.col < ce.row:
                        re = re.next_in_row
                    elif ce.row < re.col:
                        ce = ce.next_in_col
                    else:  # re.col == ce.row
                        val += re.val * ce.val
                        re = re.next_in_row
                        ce = ce.next_in_col

                if val != 0:
                    m.add_element(row, col, val)

        return m

    def mult(self, rhs):
        """ Multiply with a column vector """
        if isinstance(rhs, list):
            if len(rhs) != len(self.cols):
                raise MatrixDimError(f'Invalid rhs: length {len(rhs)} for matrix size {len(self.rows)}')
            x = rhs
        else:  # Collect a sparse rhs into a solution-list
            x = [0.0] * len(self.cols)
            if rhs is not None:
                for r, v in rhs.items(): x[r] = v

        y = [0.0] * len(self.rows)
        for (row, e) in enumerate(self.rows):
            while e is not None:
                y[row] += e.val * x[e.col]
                e = e.next_in_row

        return y

    def extract_lu(self):
        """ Return two-tuple (L,U) of matrices from factored `self` """
        assert self.state == MatrixState.FACTORED

        l = SparseMatrix()
        for e in self.diag:
            # L has unit-entries along its diagonal
            l.add_element(e.row, e.col, 1.0)
            while e.next_in_col is not None:
                e = e.next_in_col
                l.add_element(e.row, e.col, e.val)

        u = SparseMatrix()
        for e in self.diag:
            while e is not None:
                u.add_element(e.row, e.col, e.val)
                e = e.next_in_row

        return (l, u)

    def copy(self):
        """ Create an element-by-element copy """
        cp = SparseMatrix()
        for e in self.elements():
            cp.add_element(e.row, e.col, e.val)
        return cp

    def add_element(self, *args, **kwargs):
        e = Element(*args, **kwargs)
        return self.insert(e)

    @classmethod
    def identity(cls, n: int):
        m = SparseMatrix()
        for k in range(n):
            m.add_element(k, k, 1.0)
        return m


class MatrixError(Exception): pass


class MatrixDimError(MatrixError): pass


class SingularMatrix(MatrixError): pass

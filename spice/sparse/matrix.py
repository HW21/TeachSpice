from typing import Dict, List, Tuple, Union, Optional
from enum import Enum, IntEnum, auto


class Axis(Enum):
    rows = auto()
    cols = auto()

    def __invert__(self):
        if self is Axis.rows: return Axis.cols
        if self is Axis.cols: return Axis.rows
        raise ValueError


class Dir(Enum):
    down = auto()
    left = auto()


class MatrixState(IntEnum):
    CREATED = auto()
    FACTORING = auto()
    FACTORED = auto()


class AxisMapping(object):
    """ Two-vector object to help keep track of index swaps.
    Converts bi-directionally in constant time by keeping two vectors,
    `internal to external` and `external to internal`. """

    def __init__(self, size: int):
        self.e2i = list(range(size))
        self.i2e = list(range(size))
        self.history = []

    def swap_int(self, x: int, y: int):
        """ Swap internal indices x and y """
        self.i2e[x], self.i2e[y] = self.i2e[y], self.i2e[x]
        self.e2i[self.i2e[x]] = x
        self.e2i[self.i2e[y]] = y
        self.history.append((x, y))


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

    def index(self, ax: Axis):
        if ax is Axis.rows: return self.row
        if ax is Axis.cols: return self.col
        raise ValueError

    def set_index(self, ax: Axis, x: int):
        if ax is Axis.rows:
            self.row = x
        elif ax is Axis.cols:
            self.col = x
        else:
            raise ValueError

    def next(self, ax: Axis):
        if ax is Axis.rows: return self.next_in_row
        if ax is Axis.cols: return self.next_in_col
        raise ValueError

    def set_next(self, ax: Axis, e: Optional["Element"]):
        if ax is Axis.rows:
            self.next_in_row = e
        elif ax is Axis.cols:
            self.next_in_col = e
        else:
            raise ValueError


class SparseMatrix(object):
    def __init__(self):
        self.state = MatrixState.CREATED
        self.rows: List[Optional[Element]] = [None]
        self.cols: List[Optional[Element]] = [None]
        self.diag: List[Optional[Element]] = [None]
        self.fillins: List[Element] = []
        self.mappings: Optional[Dict[Axis, AxisMapping]] = None

    def elements(self, ax: Axis = Axis.cols):
        """ Iterator of elements.  Major axis set by `ax` parameter.  Default: columns. """
        for c in self.hdrs(ax):
            while c is not None:
                yield c
                c = c.next(ax)

    def values(self, *args, **kwargs):
        """ Columns-first iterator of element values"""
        for e in self.elements(*args, **kwargs): yield e.val

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
        if row == col: return self.diag[row]

        # Real search
        e = self.rows[row]
        while e is not None and e.col < col:
            e = e.next_in_row
        if e is None or e.col != col:
            return None
        MatrixError.assert_true(e.row == row)
        MatrixError.assert_true(e.col == col)
        return e

    def hdrs(self, axis: Axis):
        """ Return the axis-header array for either rows or columns. """
        MatrixError.assert_true(isinstance(axis, Axis))
        if axis is Axis.rows: return self.rows
        if axis is Axis.cols: return self.cols
        raise ValueError

    def move(self, ax: Axis, e: Element, to: int):
        """ Move element `e` to index `to` on axis `ax`, updating pointers accordingly.
        There must not be an existing element in-place.
        However there may or may not be entries above or below `to` and/or `e`. """

        # Extract element coordinates, in-axis and off-axis
        off_ax = ~ax
        idx, y = e.index(ax), e.index(off_ax)
        if idx == to: return

        if idx < to:  # Search for element first
            br = self.before_index(ax=off_ax, index=y, before=to, hint=e)

            if br is not e:  # Need some pointer-swaps
                be = self.prev(off_ax, e)
                if be is None:  # Move out of header array
                    self.hdrs(off_ax)[y] = e.next(off_ax)
                else:
                    be.set_next(off_ax, e.next(off_ax))
                e.set_next(off_ax, br.next(off_ax))
                br.set_next(off_ax, e)

        else:  # Search for row first
            br = self.before_index(ax=off_ax, index=y, before=to)
            be = self.prev(off_ax, e, hint=br)

            if br is not be:  # If we (may) need some pointer updates
                if be is not None:  # Short-circuit over `e`
                    be.set_next(off_ax, e.next(off_ax))

                if br is None:  # New first in column
                    first = self.hdrs(off_ax)[y]
                    e.set_next(off_ax, first)
                    self.hdrs(off_ax)[y] = e
                elif br is not e:  # Splice `e` in after `br`
                    e.set_next(off_ax, br.next(off_ax))
                    br.set_next(off_ax, e)

        e.set_index(ax, to)
        if idx == y:  # If `e` was on the diagonal, remove it
            self.diag[idx] = None
        elif e.row == e.col:  # Or if it's now on the diagonal, add it
            self.diag[e.row] = e

    def exchange_elements(self, ax: Axis, ex: Element, ey: Element):
        """ Swap two elements `ax` indices.
        Elements must be in the same off-axis vector,
        and the first argument `ex` must be the lower-indexed off-axis.
        E.g. exchange_elements(Axis.rows, ex, ey) exchanges the rows of ex and ey. """

        off_ax = ~ax
        MatrixError.assert_true(ex.index(off_ax) == ey.index(off_ax))
        off_idx = ey.index(off_ax)
        MatrixError.assert_true(ex.index(ax) < ey.index(ax))

        # Find the elements before each of `ex` and `ey`.
        bx = self.prev(off_ax, ex)
        by = self.prev(off_ax, ey, hint=ex)

        # Now we can get to swappin.
        tmp = ex.index(ax)
        ex.set_index(ax, ey.index(ax))
        ey.set_index(ax, tmp)

        if bx is None:  # If `ex` is the *first* entry in the column, replace it to our header-list
            self.hdrs(off_ax)[off_idx] = ey
        else:  # Otherwise patch ey into bx
            bx.set_next(off_ax, ey)

        if by is ex:  # `ex` and `ey` are adjacent
            tmp = ey.next(off_ax)
            ey.set_next(off_ax, ex)
            ex.set_next(off_ax, tmp)
        else:  # Elements in-between `ex` and `ey`.  Update the last one.
            tmp = ey.next(off_ax)
            ey.set_next(off_ax, ex.next(off_ax))
            ex.set_next(off_ax, tmp)
            by.set_next(off_ax, ex)

        if ex.row == ex.col: self.diag[ex.row] = ex
        if ey.row == ey.col: self.diag[ey.row] = ey

    def swap(self, axis: Axis, x: int, y: int):
        """ Swap either two rows or two columns, indexed `x` and `y`.
        Enum argument `axis` dictates whether to swap rows or cols. """
        if x == y: return
        if y < x: x, y = y, x

        ax_hdrs = self.hdrs(axis)
        ex = ax_hdrs[x]
        ey = ax_hdrs[y]

        while ex is not None or ey is not None:
            if ex is None:
                self.move(axis, ey, x)
                ey = ey.next(axis)
            elif ey is None or ex.index(~axis) < ey.index(~axis):
                self.move(axis, ex, y)
                ex = ex.next(axis)
            elif ey.index(~axis) < ex.index(~axis):
                self.move(axis, ey, x)
                ey = ey.next(axis)
            else:
                self.exchange_elements(axis, ex, ey)
                ex = ex.next(axis)
                ey = ey.next(axis)

        # Swap row-header pointers
        ax_hdrs[x], ax_hdrs[y] = ax_hdrs[y], ax_hdrs[x]
        # Make updates to our row-mappings
        self.mappings[axis].swap_int(x, y)

    def swap_cols(self, x: int, y: int):
        return self.swap(axis=Axis.cols, x=x, y=y)

    def swap_rows(self, x: int, y: int):
        return self.swap(axis=Axis.rows, x=x, y=y)

    def _swap_rows_deprecated(self, x: int, y: int):
        """ Hard-coded to `rows`/`cols edition """
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
        self.mappings[Axis.rows].swap_int(x, y)

    def prev(self, ax: Axis, e: Element, hint: Optional[Element] = None) -> Optional[Element]:
        """ Find the element before `e`, in direction `ax`.
        Optional hint provides an element that is as close as we know to right.
        If e is the first element in this axis, or hint is e, returns None. """
        prev = hint or self.hdrs(ax)[e.index(ax)]
        if prev is None or prev is e: return None

        nxt = prev.next(ax)
        while nxt is not None and nxt is not e:
            prev = nxt
            nxt = nxt.next(ax)
        MatrixError.assert_true(nxt is e)
        return prev

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
        MatrixError.assert_true(next is e)
        return prev

    def before_index(self, ax: Axis, index: int, before: int, hint: Optional[Element] = None) -> Optional[Element]:
        """ Find the last element in Axis `ax`, index `index`, before (off-axis) index `before`.
        E.g. before_index(Axis.cols, 3, 7) would be the element in column 3 before row 7.
        Optional `hint` is an element *known* to lead to the answer, through calls to `next(axis)`.
        If no hint is provided, search starts from the Axis-headers. """
        off_ax = ~ax
        prev = hint or self.hdrs(ax)[index]
        if prev is None or prev.index(off_ax) > before: return None

        nxt = prev.next(ax)
        while nxt is not None and nxt.index(off_ax) < before:
            prev = nxt
            nxt = nxt.next(ax)
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
        MatrixError.assert_true(ex.col == ey.col)
        MatrixError.assert_true(ex.row < ey.row)

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
        """ Insert new Element `e` """
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
        elif col_head.row > e.row:
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

        # Quick singularity check
        # Each row & column must have entries
        for e in self.cols:
            if e is None: raise SingularMatrix
        for e in self.rows:
            if e is None: raise SingularMatrix

        self.set_state(MatrixState.FACTORING)

        for n, d in enumerate(self.diag[:-1]):
            # Find a pivot element
            pivot = self.find_max(n)
            # Swap it onto our diagonal at index `n`
            self.swap(Axis.rows, pivot.row, n)
            self.swap(Axis.cols, pivot.col, n)
            MatrixError.assert_true(self.diag[n] is pivot)
            # And convert its row/column
            self.row_col_elim(pivot)

        self.set_state(MatrixState.FACTORED)

    def find_max(self, n: int = 0) -> Element:
        """ Find the max (abs value) element in sub-matrix of indices ≥ `n`. """

        e = self.diag[n]  # Start from the diagonal, if available
        if e is None:
            e = self.cols[n]
            while e.row < n:
                e = e.next_in_col
        max_elem = self.find_max_below(e)

        for e in self.cols[n:]:
            while e.row < n:
                e = e.next_in_col
            max_in_col = self.find_max_below(e)
            if abs(max_in_col.val) > abs(max_elem.val):
                max_elem = max_in_col
        return max_elem

    def find_max_below(self, below: Element) -> Element:
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
        MatrixError.assert_true(self.state == MatrixState.FACTORED)

        if isinstance(rhs, list):
            MatrixError.assert_eq(len(rhs), len(self.rows))
            ##, f'Invalid rhs: length {len(rhs)} for matrix size {len(self.rows)}'
            c = rhs[:]
            for n, v in enumerate(rhs):
                c[self.mappings[Axis.rows].e2i[n]] = v
        elif rhs is not None:  # Collect a sparse rhs into a dense list
            c = [0.0] * len(self.rows)
            for r, v in rhs.items():
                c[self.mappings[Axis.rows].e2i[r]] = v

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

        # Unwind any column-swaps
        soln = [0.0] * len(c)
        for k in range(len(c)):
            soln[k] = c[self.mappings[Axis.cols].e2i[k]]
        return soln

    def matmul(self, other):
        """ Matrix multiplication self*other """
        if len(self.rows) != len(other.cols):
            raise MatrixDimError

        m = SparseMatrix()
        for (row, re) in enumerate(self.rows):
            for (col, ce) in enumerate(other.cols):
                # "Two pointer" row * col dot-product
                val = 0
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
        MatrixError.assert_true(self.state == MatrixState.FACTORED)

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

    def set_state(self, state: MatrixState):
        if state is MatrixState.FACTORING:
            MatrixError.assert_true(self.state is MatrixState.CREATED)
            self.mappings = {
                Axis.rows: AxisMapping(len(self.rows)),
                Axis.cols: AxisMapping(len(self.cols)),
            }
            self.state = state
        elif state is MatrixState.FACTORED:
            MatrixError.assert_true(self.state is MatrixState.FACTORING)
            self.state = state
        else:
            raise ValueError

    def _checkup(self):
        """ Internal consistency tests.  Probably pretty slow. """

        next_in_cols = list()
        next_in_rows = list()
        for n, e in enumerate(self.cols):
            while e is not None:
                MatrixError.assert_eq(e.col, n)
                if e.next_in_col is not None:
                    MatrixError.assert_true(e.next_in_col.row > e.row)
                    MatrixError.assert_true(e.next_in_col not in next_in_cols)
                    next_in_cols.append(e.next_in_col)
                if e.next_in_row is not None:
                    MatrixError.assert_true(e.next_in_row.col > e.col)
                    MatrixError.assert_true(e.next_in_row not in next_in_rows)
                    next_in_rows.append(e.next_in_row)
                e = e.next_in_col

        next_in_cols = list()
        next_in_rows = list()
        for n, e in enumerate(self.rows):
            while e is not None:
                MatrixError.assert_eq(e.row, n)
                if e.next_in_col is not None:
                    MatrixError.assert_true(e.next_in_col.row > e.row)
                    MatrixError.assert_true(e.next_in_col not in next_in_cols)
                    next_in_cols.append(e.next_in_col)
                if e.next_in_row is not None:
                    MatrixError.assert_true(e.next_in_row.col > e.col)
                    MatrixError.assert_true(e.next_in_row not in next_in_rows)
                    next_in_rows.append(e.next_in_row)
                e = e.next_in_row

        for n, e in enumerate(self.diag):
            MatrixError.assert_is(self.diag[n], self.get(n, n))


class MatrixError(Exception):
    @classmethod
    def assert_true(cls, cond):
        if not cond:
            raise cls

    @classmethod
    def assert_eq(cls, x, y):
        if x != y:
            raise cls

    @classmethod
    def assert_is(cls, x, y):
        if x is not y:
            raise cls


class MatrixDimError(MatrixError): pass


class SingularMatrix(MatrixError): pass

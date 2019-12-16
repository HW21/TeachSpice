from typing import Dict, List, Union, Optional
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


class MatrixState(IntEnum):
    CREATED = auto()
    FACTORING = auto()
    FACTORED = auto()


class SparseMatrix(object):
    def __init__(self):
        self.rows: List[Optional[Element]] = [None]
        self.cols: List[Optional[Element]] = [None]
        self.diag: List[Optional[Element]] = [None]
        self.state = MatrixState.CREATED

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
        if row > len(self.rows) - 1: return None
        if col > len(self.cols) - 1: return None

        # Easy access cases
        if row == 0: return self.cols[col]
        if col == 0: return self.rows[row]
        # if row == col: return self.diag[row] # FIXME

        # Real search
        e = self.rows[row]
        while e is not None and e.col < col:
            e = e.next_in_row
        if e is None or e.row != row:
            return None
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

        # Swap row-head pointers
        self.rows[x], self.rows[y] = self.rows[y], self.rows[x]
        # FIXME: keep track of swaps for RHS

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

    def exchange_col_elements(self, ex: Element, ey: Element):
        """ Swap the rows of two elements in the same column """
        assert ex.col == ey.col
        assert ex.row < ey.row
        bx = self.cols[ex.col]
        assert bx is not None

        if bx is not ex:
            # Find element before `ex` in the column
            next = bx.next_in_col
            while next is not None and next.row < ex.row:
                bx = next
                next = next.next_in_col
            assert next is ex

        # Now carry along to find the element before `ey`
        by = ex
        next = by.next_in_col
        while next is not None and next.row < ey.row:
            by = next
            next = next.next_in_col
        assert next is ey

        # At this point,
        # * bx is either the element before ex, or ex itself
        # * by is the element before ey, which may be ex
        # Get it?

        # Now we can get to swappin.
        if bx is ex:  # If `ex` is the *first* entry in the column, replace it to our list
            self.cols[ex.col] = ey
        else:  # Otherwise patch ey into bx
            bx.next_in_col = ey
        if by is not ex:  # If there are any elements in-between `ex` and `ey`, update the last one
            by.next_in_col = ex

        ex.next_in_col, ey.next_in_col = ey.next_in_col, ex.next_in_col
        ex.row, ey.row = ey.row, ex.row

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
        for pivot in self.diag:
            self.row_col_elim(pivot)
        self.state = MatrixState.FACTORED

    def row_col_elim(self, pivot: Element):
        """ Eliminate the row & column of element `pivot`,
        transforming them into the LU-factored row/col of L and U. """
        if pivot.val == 0: raise SingularMatrix

        pupper = pivot.next_in_row
        while pupper is not None:
            plower = pivot.next_in_col
            # pabove = pupper
            psub = pupper.next_in_col
            while plower is not None:
                # row = plower.row
                # val = plower.val
                plower.val /= pivot.val
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

        if isinstance(rhs, list):
            assert len(rhs) == len(self.rows), f'Invalid rhs: length {len(rhs)} for matrix size {len(self.rows)}'
            c = rhs
        else: # Collect a sparse rhs into a solution-list
            c = [0.0] * len(self.rows)
            if rhs is not None:
                for r, v in rhs.items(): c[r] = v

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

    def add_element(self, *args, **kwargs):
        e = Element(*args, **kwargs)
        return self.insert(e)


class SingularMatrix(Exception): pass

import pytest
from ..matrix import Element, SparseMatrix, MatrixError, MatrixState


def test_create_element():
    e = Element(row=0, col=0, val=1.0)
    assert e.row == 0
    assert e.col == 0
    assert e.val == 1.0


def test_create_matrix():
    m = SparseMatrix()
    assert m.rows == [None]
    assert m.cols == [None]
    assert m.diag == [None]


def test_add_elements():
    m = SparseMatrix()

    e = m.add_element(0, 0, 1.0)
    assert len(m.rows) == 1
    assert len(m.cols) == 1
    assert len(m.diag) == 1
    assert m.rows[0] is e
    assert m.cols[0] is e
    assert m.diag[0] is e

    e = m.add_element(100, 100, 1.0)
    assert len(m.rows) == 101
    assert len(m.cols) == 101
    assert len(m.diag) == 101
    assert m.rows[100] is e
    assert m.cols[100] is e
    assert m.diag[100] is e


def test_get():
    m = SparseMatrix()
    m.add_element(0, 0, 1.0)
    e = m.get(0, 0)
    assert e is not None
    assert e.val == 1.0
    m.add_element(10, 10, -1.0)
    e = m.get(10, 10)
    assert e is not None
    assert e.val == -1.0


def test_swap_rows():
    m = SparseMatrix()

    m.add_element(0, 0, 11.0)
    m.add_element(7, 0, 22.0)
    m.add_element(0, 7, 33.0)
    m.add_element(7, 7, 44.0)

    # Fake-set the FACTORING state
    m.state = MatrixState.FACTORING
    m.rowmap_e2i = list(range(len(m.rows)))
    m.rowmap_i2e = list(range(len(m.rows)))

    m.swap_rows(0, 7)

    e = m.get(7, 0)
    assert e is not None
    assert e.val == 11.0
    e = m.get(0, 7)
    assert e is not None
    assert e.val == 44.0

    m = SparseMatrix()

    m.add_element(0, 0, 1)
    m.add_element(0, 1, 2)
    m.add_element(0, 2, 3)
    m.add_element(1, 0, 4)
    m.add_element(1, 1, 5)
    m.add_element(1, 2, 6)
    m.add_element(2, 0, 7)
    m.add_element(2, 1, 8)
    m.add_element(2, 2, 9)

    # Fake-set the FACTORING state
    m.state = MatrixState.FACTORING
    m.rowmap_e2i = list(range(len(m.rows)))
    m.rowmap_i2e = list(range(len(m.rows)))

    m.swap_rows(0, 2)
    e = m.get(0, 0)
    assert e is not None
    assert e.val == 7.0

    m = SparseMatrix()
    m.add_element(1, 0, 71)
    m.add_element(2, 0, -11)
    m.add_element(2, 2, 99)

    # Fake-set the FACTORING state
    m.state = MatrixState.FACTORING
    m.rowmap_e2i = list(range(len(m.rows)))
    m.rowmap_i2e = list(range(len(m.rows)))

    m.swap_rows(0, 2)

    e = m.get(0, 0)
    assert e is not None
    assert e.val == -11


def test_row_mappings():
    m = SparseMatrix.identity(4)

    # Fake-set the FACTORING state
    m.state = MatrixState.FACTORING
    m.rowmap_e2i = list(range(len(m.rows)))
    m.rowmap_i2e = list(range(len(m.rows)))

    m.swap_rows(0, 3)
    assert m.rowmap_e2i == [3, 1, 2, 0]
    assert m.rowmap_i2e == [3, 1, 2, 0]

    m.swap_rows(0, 2)
    assert m.rowmap_e2i == [3, 1, 0, 2]
    assert m.rowmap_i2e == [2, 1, 3, 0]


def test_eq():
    m = SparseMatrix()
    m.add_element(0, 0, 1.0)
    m.add_element(1, 1, 1.0)
    m.add_element(2, 2, 1.0)

    m1 = SparseMatrix()
    m1.add_element(0, 0, 1.0)
    m1.add_element(1, 1, 1.0)
    m1.add_element(2, 2, 1.0)

    assert m == m1

    m2 = SparseMatrix()
    m2.add_element(0, 0, 1.0)
    m2.add_element(1, 1, 1.0)
    m2.add_element(2, 2, 1.0)
    m2.add_element(3, 3, 1.0)

    assert m != m2


def test_lu_id3():
    m = SparseMatrix()
    m.add_element(0, 0, 1.0)
    m.add_element(1, 1, 1.0)
    m.add_element(2, 2, 1.0)

    check_diagonal(m)

    m.lu_factorize()

    check_diagonal(m)

    e = m.get(0, 0)
    assert e.val == 1.0
    e = m.get(1, 1)
    assert e.val == 1.0
    e = m.get(2, 2)
    assert e.val == 1.0


def test_lu_lower():
    m = SparseMatrix()
    m.add_element(0, 0, 1.0)
    m.add_element(1, 0, 1.0)
    m.add_element(2, 0, 1.0)
    m.add_element(1, 1, 1.0)
    m.add_element(2, 1, 1.0)
    m.add_element(2, 2, 1.0)

    check_diagonal(m)

    m.lu_factorize()

    check_diagonal(m)

    e = m.get(0, 0)
    assert e.val == 1.0
    e = m.get(1, 1)
    assert e.val == 1.0
    e = m.get(2, 2)
    assert e.val == 1.0


def test_swap():
    m = SparseMatrix()
    m.add_element(0, 0, 1.0)
    m.add_element(0, 1, 1.0)
    m.add_element(0, 2, 1.0)
    m.add_element(1, 1, 2.0)
    m.add_element(1, 2, 5.0)
    m.add_element(2, 0, 2.0)
    m.add_element(2, 1, 5.0)
    m.add_element(2, 2, -1.0)

    # Fake-set the FACTORING state
    m.state = MatrixState.FACTORING
    m.rowmap_e2i = list(range(len(m.rows)))
    m.rowmap_i2e = list(range(len(m.rows)))

    m.swap_rows(0, 2)

    e1 = m.get(0, 0)
    assert e1.val == 2.0
    e2 = m.get(2, 0)
    assert e2.val == 1.0
    assert e1.next_in_col is e2
    assert e2.next_in_col is None

    e01 = m.get(0, 1)
    e11 = m.get(1, 1)
    e21 = m.get(2, 1)
    assert e01.next_in_col is e11
    assert e11.next_in_col is e21
    assert e21.next_in_col is None


def test_lu():
    m = SparseMatrix()
    m.add_element(0, 0, 1.0)
    m.add_element(0, 1, 1.0)
    m.add_element(0, 2, 1.0)
    m.add_element(1, 1, 2.0)
    m.add_element(1, 2, 5.0)
    m.add_element(2, 0, 2.0)
    m.add_element(2, 1, 5.0)
    m.add_element(2, 2, -1.0)

    check_diagonal(m)

    m.lu_factorize()

    check_diagonal(m)


def test_solve():
    m = SparseMatrix()

    m.add_element(0, 0, 1.0)
    m.add_element(0, 1, 1.0)
    m.add_element(0, 2, 1.0)
    m.add_element(1, 1, 2.0)
    m.add_element(1, 2, 5.0)
    m.add_element(2, 0, 2.0)
    m.add_element(2, 1, 5.0)
    m.add_element(2, 2, -1.0)

    check_diagonal(m)
    m2 = m.copy()

    m.lu_factorize()
    check_diagonal(m)

    l, u = m.extract_lu()
    m3 = l.matmul(u)
    print(m3)

    c = m.solve(rhs={0: 6, 1: -4, 2: 27})
    assert c == [5, 3, -2]


def check_diagonal(m: SparseMatrix):
    """ Helper function.
    Check that the diagonal and get(r,r) are consistent. """
    for r in range(len(m.diag)):
        e = m.get(r, r)
        assert e is m.diag[r]


def test_identity():
    i2 = SparseMatrix.identity(2)
    assert len(i2.rows) == 2
    assert len(i2.cols) == 2
    e = i2.get(0, 0)
    assert e.val == 1
    assert i2.diag[0] is e
    assert e.next_in_col is None
    assert e.next_in_row is None
    e = i2.get(1, 1)
    assert e.val == 1


def test_mult():
    i3 = SparseMatrix.identity(3)
    y = i3.mult([3, 4, 5])
    assert y == [3, 4, 5]

    with pytest.raises(MatrixError):
        i3.mult([6, 7, 8, 9])


def test_extract_lu():
    i4 = SparseMatrix.identity(4)
    i4.lu_factorize()
    l, u = i4.extract_lu()
    assert l == i4
    assert u == i4


def test_copy():
    i9 = SparseMatrix.identity(9)
    cp = i9.copy()
    assert cp == i9
    assert cp is not i9


def test_matmul():
    i5a = SparseMatrix.identity(5)
    i5b = SparseMatrix.identity(5)
    assert i5a == i5b
    assert i5a is not i5b
    i5c = i5a.matmul(i5b)
    assert i5a == i5c
    assert i5a is not i5c
    assert i5b == i5c
    assert i5b is not i5c

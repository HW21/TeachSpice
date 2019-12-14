from .. import Element, SparseMatrix


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

    m.swap_rows(0, 2)
    e = m.get(0, 0)
    assert e is not None
    assert e.val == 7.0

    m = SparseMatrix()
    m.add_element(1, 0, 71)
    m.add_element(2, 0, -11)
    m.add_element(2, 2, 99)

    m.swap_rows(0, 2)

    e = m.get(0, 0)
    assert e is not None
    assert e.val == -11

import math
from pathlib import Path
from spice.sparse.yaml import MatrixYaml

SIZE_LIMIT = None


def eq_or_close(xi, yi):
    """ Set our params for `math.isclose` """
    return math.isclose(xi, yi, rel_tol=1e-6, abs_tol=1e-9)


def eq(x: list, y: list) -> bool:
    """ Compensate for limited precision saved in file outputs """
    for xi, yi in zip(x, y):
        if not eq_or_close(xi, yi): return False
    return True


def print_vec_compare(x: list, y: list):
    """ Print a comparison table between two vectors (lists) """
    print(f"      x      y      diff     isclose")
    for xi, yi in zip(x, y):
        s = f'{xi: 7e} {yi: 7e} {abs(xi - yi): 7e}  '
        s += str(eq_or_close(xi, yi))
        print(s)


def peek_size(path):
    """ Take a quick peek at the `size` field of our YAML schema.
    One (hefty) downside of using YAML: parsers read all at once.
    So big files can be pretty, pretty slow. """
    with open(path, 'r')as f:
        _ = f.readline()  # Line 1: Description
        size_line = f.readline()  # Line 2: Size
        if "size" not in size_line:
            raise Exception
        _, size_str = size_line.strip().split()
        size = int(size_str)
        return size


def run_case(p: Path, res: dict):
    print(f"Running Test-Case {p.name}")

    try:  # only run small cases, for now
        size = peek_size(p)
        res['size'] = size
        if SIZE_LIMIT is not None and size > SIZE_LIMIT:
            print("Size over limit, skipping")
            return
    except Exception as e:
        print(e)
        return

    try:
        print(f"Reading YAML")
        y = MatrixYaml.load(p)
        print(f"Converting")
        m = y.to_mat()
    except Exception as e:
        return
    else:
        res['read'] = True
        res['rhs'] = y.rhs is not None

    # z = m.mult(y.solution)
    # print_vec_compare(y.rhs, z)

    try:
        print(f"Factorizing")
        m.lu_factorize()
    except Exception as e:
        print(e)
        return
    else:
        res['factor'] = True

    # Skip solving the no-RHS cases; havent really figured out what to make of them
    if y.rhs is None:
        return

    try:
        print(f"Solving")
        x = m.solve(rhs=y.rhs)
    except Exception as e:
        print(e)
        return
    else:
        res['solve'] = True
        correct = eq(x, y.solution)
        res['correct'] = correct
        if correct:
            print(f"Test Case Succeeded: {p.name}")
        else:
            print(f"Incorrect Solution")
            print_vec_compare(x, y.solution)

    try:  # Check by multiplying with our calculated solution
        y = MatrixYaml.load(p)
        m2 = y.to_mat()
        z = m2.mult(x)
    except Exception as e:
        print(e)
        return
    else:
        if y.rhs:
            match = eq(z, y.rhs)
            res['mult'] = match

            if match:
                print(f"Mult-back matched RHS: {p.name}")
            else:
                print(f"Mult-back yield incorrect: {x}")
                print(f"Correct RHS: {y.rhs}")
                print_vec_compare(z, y.rhs)


def yaml_testcases():
    """ Run all YAML matrices in data/ dir """
    run_all = True
    if run_all:
        paths = Path("data/").glob("*.yaml")
    else:
        paths = Path("data/").glob("sptrMondo.tr.mat.yaml")

    results = []
    for p in paths:
        res = dict(
            file=p.name,
            size=None,
            read=False,
            factor=False,
            rhs=False,
            solve=False,
            correct=False,
            mult=False,
        )
        results.append(res)
        run_case(p, res)

    # Print some summary info
    summary = 'File'.ljust(30)
    for k in results[0]:
        if k != 'file': summary += k.ljust(7)
    summary += '\n'

    for r in sorted(results, key=lambda r: r['file']):
        s = r.pop('file').ljust(30)
        for k in r: s += str(r[k]).ljust(7)
        summary += s + '\n'

    print(summary)
    if run_all:  # Save status to file
        with open('testcases.txt', 'w') as f:
            f.write(summary)


if __name__ == '__main__':
    yaml_testcases()

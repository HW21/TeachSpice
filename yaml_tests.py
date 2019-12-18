import math
from pathlib import Path
from spice.sparse.yaml import MatrixYaml


def eq_or_close(xi, yi):
    """ Set our params for `math.isclose` """
    return math.isclose(xi, yi, rel_tol=1e-6, abs_tol=1e-9)


def eq(x: list, y: list) -> bool:
    """ Compensate for limited precision saved in file outputs """
    for xi, yi in zip(x, y):
        if not eq_or_close(xi, yi): return False
    return True


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


def yaml_testcases():
    """ Run all YAML matrices in data/ dir """

    results = []
    for p in Path("data/").glob("*.yaml"):
        print(f"Running Test-Case {p.name}")

        res = dict(
            file=p.name,
            size=None,
            read=False,
            rhs=False,
            factor=False,
            solve=False,
            correct=False
        )
        results.append(res)

        try:  # only run small cases, for now
            size = peek_size(p)
            res['size'] = size
            if size > 1000:
                continue
        except Exception as e:
            print(e)
            continue

        try:
            print(f"Reading YAML")
            y = MatrixYaml.load(p)
            print(f"Converting")
            m = y.to_mat()
        except Exception as e:
            continue
        else:
            res['read'] = True
            res['rhs'] = y.rhs is not None

        try:
            print(f"Factorizing")
            m.lu_factorize()
        except Exception as e:
            print(e)
            continue
        else:
            res['factor'] = True

        try:
            print(f"Solving")
            x = m.solve(rhs=y.rhs)
        except Exception as e:
            print(e)
            continue
        else:
            res['solve'] = True
            correct = eq(x, y.solution)
            res['correct'] = correct
            if correct:
                print(f"Test Case Succeeded: {p.name}")
            else:
                print(f"Incorrect Solution")
                print(f"      x      y      diff     isclose")
                for xi, yi in zip(x, y.solution):
                    s = f'{xi: 7e} {yi: 7e} {abs(xi - yi): 7e}  '
                    s += str(eq_or_close(xi, yi))
                    print(s)

        try:  # Check by multiplying with our calculated solution
            y = MatrixYaml.load(p)
            m2 = y.to_mat()
            z = m2.mult(x)
        except Exception as e:
            print(e)
            continue
        else:
            if y.rhs:
                match = eq(z, y.rhs)
                res['mult'] = match

                if match:
                    print(f"Mult-back matched RHS: {p.name}")
                else:
                    print(f"Mult-back yield incorrect: {x}")
                    print(f"Correct RHS: {y.rhs}")

    print(f"{'File'.ljust(30)}Size   Read   RHS    Factor Solve  Correct")

    def fmt(b) -> str:
        return str(b).ljust(7)

    for r in sorted(results, key=lambda r: r['file']):
        print(f"{r['file'].ljust(30)}{fmt(r['size'])}" +
              f"{fmt(r['read'])}{fmt(r['read'])}{fmt(r['factor'])}{fmt(r['solve'])}{fmt(r['correct'])}")


if __name__ == '__main__':
    yaml_testcases()

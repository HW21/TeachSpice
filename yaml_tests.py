from pathlib import Path
from spice.sparse.yaml import MatrixYaml


def yaml_testcases():
    """ Run all YAML matrices in data/ dir """

    results = []
    for p in Path("data/").glob("*.yaml"):
        print(f"Running Test-Case {p.name}")

        res = dict(
            file=p.name,
            read=False,
            factor=False,
            solve=False,
            correct=False
        )
        results.append(res)

        try:
            print(f"Reading YAML")
            y = MatrixYaml.load(p)
            if y.size > 10:
                continue
            print(f"Converting")
            m = y.to_mat()
        except Exception as e:
            continue
        else:
            res['read'] = True

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
            correct = x == y.solution
            res['correct'] = correct
            if correct:
                print(f"Test Case Succeeded: {p.name}")
            else:
                print(f"Incorrect Solution: {x}")
                print(f"Solution (from YAML): {y.solution}")

    print(f"{'File'.ljust(30)}Read   Factor Solve  Correct")

    def fmt(b: bool) -> str:
        return str(b).ljust(7)

    for r in results:
        print(f"{r['file'].ljust(30)}{fmt(r['read'])}{fmt(r['factor'])}{fmt(r['solve'])}{fmt(r['correct'])}")


if __name__ == '__main__':
    yaml_testcases()

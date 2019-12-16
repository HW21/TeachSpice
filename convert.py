"""
Convert example matrices from the Kundert `sparse` project
"""

import os
from pathlib import Path

from spice.sparse.file import SparseFile
from spice.sparse.yaml import MatrixYaml


def convert_sparse_to_yaml():
    """ Convert all *.mat files to YAML """
    paths = []
    results = []
    for path in Path(os.environ['SPARSE_DIR']).glob("matrices/*.mat"):
        print(f"Reading {path}")
        paths.append(path)
        res = dict(
            real=False,
            read=False,
            rhs=False,
            run=False,
            yaml=False,
            convert=False,
            factor=False,
        )
        results.append(res)

        sf = None
        try:
            sf = SparseFile(path)
            sf.read()
        except Exception as e:
            res["real"] = False
            continue
        else:
            res["real"] = True
            res["read"] = True
            res["rhs"] = sf.rhs is not None

        soln = None
        try:
            soln = run_sparse(path)
        except Exception as e:
            print(e)
            res["run"] = False
            continue
        else:
            res["run"] = True

        yaml = None
        try:
            yaml = MatrixYaml.from_sparse_file(sf)
            yaml.solution = soln

            fname = Path(path).name
            yaml.dump(f"data/{fname}.yaml")
        except AssertionError as e:
            # ruamel.yaml seems to throw these on working cases
            print(f"Another Mystery YAML Assertion Error")
            print(e)
            res["yaml"] = True
        except Exception as e:
            print(e)
            res["yaml"] = False
            continue
        else:
            res["yaml"] = True

    for path, res in zip(paths, results):
        print(f"{path.ljust(50)} : {res}")


def run_sparse(filename):
    sparse = os.environ['SPARSE_DIR'] + '/bin/sparse'

    import subprocess
    print("Launching Sparse")
    result = subprocess.run([sparse, "-s", filename], stdout=subprocess.PIPE)
    print("Sparse Done")

    s = result.stdout.decode("utf-8")

    print("Parsing Solution")
    soln = []
    state = 'header'
    for line in s.splitlines():
        if state == 'header':
            if not line.strip():
                print(line)
                state = 'soln'
        elif state == 'soln':
            if not line:
                break
            soln.append(float(line.strip()))
    if not soln:
        print(f"No solution found for {filename}")
    return soln


if __name__ == '__main__':
    convert_sparse_to_yaml()

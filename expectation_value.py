from qiskit.result.result import Result

from w import OrderedOperator


def projective_measurement(result: Result, operator: OrderedOperator) -> float:
    if len(result.results) != 1:
        raise RuntimeError("More than one experiment is present.")

    shots: int = result.results[0].shots
    counts = result.get_counts()

    # validate the counts dict
    if any(len(key) != operator.N for key in counts):
        raise ValueError("Malformed key in counts dict")
    if sum(counts.values()) != shots:
        raise ValueError("The entries of the counts dict doesn't sum up to the number of shots.")

    # all possible measurement outcomes and thereby all possible keys of the counts dict
    #keys = [format(key, f'0{operator.N}b') for key in range(2**operator.N)]
    #print("\t\t".join(f"{key}:{operator.sign(key)}{counts.get(key, 0)}" for key in keys), "\t\t", operator)

    return sum(operator.sign(key) * counts[key] for key in counts) / shots

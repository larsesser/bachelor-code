from qiskit.result.result import Result

from w import OrderedOperator


def expectation_value(result: Result, operator: OrderedOperator) -> float:
    """Calculate the expectation value from the given operator.

    This assumes that measurement was done in the computational basis.
    """
    if len(result.results) != 1:
        raise RuntimeError("More than one experiment is present.")

    # the overall number of times the state was prepared and measured
    shots: int = result.results[0].shots
    counts = result.get_counts()

    # validate the counts dict
    if any(len(key) != operator.N for key in counts):
        raise ValueError("Malformed key in counts dict")
    if sum(counts.values()) != shots:
        raise ValueError("The entries of the counts dict doesn't sum up to the"
                         " number of shots.")

    return sum(operator.sign(key) * counts[key] for key in counts) / shots

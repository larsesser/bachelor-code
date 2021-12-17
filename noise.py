from typing import Dict, List
from collections import defaultdict
from sympy import S, Symbol

from qiskit.providers.aer.noise import NoiseModel, ReadoutError

from w import unravel_symbol

# Mapping a symbol retrieved from w.py p0_symbol / p1_symbol to the flipping probability
ErrorProbabilities = Dict[Symbol, float]


def add_equal_qubit_readout_errors(noise: NoiseModel, error: float, qubits: int):
    error = S(error)
    combinations = 2 ** qubits
    errors = [[0 for _ in range(combinations)] for _ in range(combinations)]
    for x in range(combinations):
        for y in range(combinations):
            # count the places where x and y differ, using the bitwise xor and count the 1s
            differences = bin(x ^ y).count("1")
            errors[x][y] = (error ** differences) * ((1 - error) ** (qubits - differences))
    readout_error = ReadoutError(errors)
    print(readout_error)
    noise.add_readout_error(readout_error, qubits=list(range(qubits)))
    print(noise)
    return noise


# TODO tatsächlich für alle qubits gleichzeitig den readout error angegeben
#  sowas wie [[(1-p)^2, (1-p)*p, p*(1-p), p^2], [(1-p)*p, (1-p)^2, p^2, (1-p)*p] etc
def add_single_qubit_readout_errors(noise: NoiseModel, errors: ErrorProbabilities) -> NoiseModel:
    """Add single qubit readout errors to each qubit as defined in errors.

    The correction procedure assumes that each qubit flips independently of the other
    ones during readout.
    """
    qubit_error: Dict[int, List[List[float, float], List[float, float]]] = defaultdict(list)
    # sort the errors by qubit
    for symbol, probability in errors.items():
        bitflip_infos = unravel_symbol(symbol)
        qubit = bitflip_infos.qubit
        from_state = bitflip_infos.from_state
        # construct the error after schema [P(noisy|ideal), P(noisy|ideal)]
        if from_state == 0:
            error = [1-probability, probability]
        else:
            error = [probability, 1-probability]
        # respect the order of errors (first |0> -> |1> error, then |1> -> |0> error)
        qubit_error[qubit].insert(from_state, error)

    for qubit, error in qubit_error.items():
        if len(error) != 2:
            raise ValueError(f"Only one type of error was provided for qubit {qubit}.")
        noise.add_readout_error(error=ReadoutError(error), qubits=[qubit])

    return noise

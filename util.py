import statistics
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from tabulate import tabulate

# prevent cyclic import, BenchmarkResult is only needed for type annotation
if TYPE_CHECKING:
    from test_mitigation import BenchmarkResult
else:
    BenchmarkResult = Any


AnglePair = Tuple[float, float]


def random_angles(N: int) -> List[AnglePair]:
    """Draw a pair of angles from [0, 2*pi) for each qubit."""
    return [
        (
            np.random.uniform(low=0, high=2 * np.pi),
            np.random.uniform(low=0, high=2 * np.pi)
        )
        for _ in range(N)
    ]


def init_random_state(N: int, angles: List[AnglePair]) -> QuantumCircuit:
    """Returns a QuantumCircuit initialized in a random state.

    A RX(sigma) and RZ(theta) is applied to each qubit, where sigma and theta
    are angles from [0, 2*pi). They are stored in the angles' parameter.

    To include entangled states, a CX gate is applied to each combination of
    control- and target qubits.
    """
    if N != len(angles):
        raise ValueError(f"Must provide equal number of qubits ({N}) and pairs"
                         f" of angles ({len(angles)}).")

    circ = QuantumCircuit(N)

    # prepare each qubit in an arbitrary state on the bloch sphere
    for bit, (sigma, theta) in enumerate(angles):
        circ.rx(sigma, bit)
        circ.rz(theta, bit)

    # entangle the qubits
    for bit1 in range(N):
        for bit2 in range(N):
            if bit1 == bit2:
                continue
            circ.cnot(bit1, bit2)

    return circ


def print_result(results: List[BenchmarkResult]):
    """Print some statistics for the given results."""
    noisy = [result.noisy for result in results]
    noiseless = [result.noiseless for result in results]
    corrected = [result.corrected for result in results]

    noisy_compairson = [
        abs(noisy_element - noiseless_element) / abs(noiseless_element)
        for noisy_element, noiseless_element in zip(noisy, noiseless)
    ]
    corrected_compairson = [
        abs(corrected_element - noiseless_element) / abs(noiseless_element)
        for corrected_element, noiseless_element in zip(corrected, noiseless)
    ]

    is_corrected_larger = [
        "!" if c > n else ""
        for c, n in zip(corrected_compairson, noisy_compairson)
    ]

    print(
        tabulate(
            zip(noiseless, noisy, corrected, noisy_compairson,
                corrected_compairson, is_corrected_larger),
            headers=["Noiseless", "Noisy", "Corrected",
                     "|(Noisy-Noiseless)|/|Noiseless|",
                     "|(Corrected-Noiseless)|/|Noiseless|", "C>N"]
        )
    )

    print(
        tabulate(
            [
                ["mean", statistics.mean(noisy_compairson),
                 statistics.mean(corrected_compairson)],
                ["stdev", statistics.stdev(noisy_compairson),
                 statistics.stdev(corrected_compairson)],
                ["median", statistics.median(noisy_compairson),
                 statistics.median(corrected_compairson)]
            ],
            headers=["", "Noisy", "Corrected"]
        )
    )

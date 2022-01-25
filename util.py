import statistics
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from tabulate import tabulate

if TYPE_CHECKING:
    from test_mitigation import TestResult
else:
    TestResult = Any


AnglePair = Tuple[float, float]


def random_angles(n_qubits: int) -> List[AnglePair]:
    """Draws a pair of angles from [0, 2*pi) for each Qubit."""
    return [
        (
            np.random.uniform(low=0, high=2 * np.pi),
            np.random.uniform(low=0, high=2 * np.pi)
        )
        for _ in range(n_qubits)
    ]


def init_random_state(qubits: int, angles: List[Tuple[float, float]]) -> QuantumCircuit:
    """Returns a QuantumCircuit where each qubit is initialized in a random state.

    The random state is realised through an RX and RZ gate acting on each qubit and
    using the angles provided in angles[qubit]. To realise entangled states, a CX gate
    is applied to each forwarding combination of qubits.
    """
    if qubits != len(angles):
        raise ValueError(f"Must provide equal number of qubits ({qubits}) and pairs of"
                         f"angles ({len(angles)}).")

    circ = QuantumCircuit(qubits)

    # prepare each qubit in an arbitrary state on the bloch sphere
    for bit, (sigma, theta) in enumerate(angles):
        circ.rx(sigma, bit)
        circ.rz(theta, bit)

    # entangle the qubits
    for bit1 in range(qubits):
        for bit2 in range(qubits):
            if bit1 == bit2:
                continue
            circ.cnot(bit1, bit2)

    return circ


def print_result(results: List[TestResult]):
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
        "!" if corrected_element > noisy_element else ""
        for corrected_element, noisy_element in zip(corrected_compairson, noisy_compairson)
    ]

    print(
        tabulate(
            zip(noiseless, noisy, corrected, noisy_compairson, corrected_compairson,
                is_corrected_larger),
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

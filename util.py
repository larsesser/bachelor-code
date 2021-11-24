import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, List, Set

import numpy as np
from time import time

from qiskit import QuantumCircuit
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.models.backendconfiguration import QasmBackendConfiguration
from qiskit.circuit.gate import Gate
from qiskit.result.result import Result
from qiskit.result.models import ExperimentResult
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city, plot_circuit_layout
from qiskit.quantum_info import Operator
from qiskit.circuit import Parameter
from qiskit.circuit.instruction import Instruction
from qiskit.algorithms.phase_estimators.phase_estimation import PhaseEstimation
from qiskit.algorithms.phase_estimators.phase_estimation_result import PhaseEstimationResult

from pprint import pprint


class LIModel:
    # factor for the Zq Zq+1 term
    J = 0   # J < 0
    # factor for the Zq term
    h = 0   # h > 0
    # number of qubits
    N = 0

    def __init__(self, *, J=0, h=0, N=0):
        self.J = J
        self.h = h
        self.N = N

    def _operator(self, time: float) -> Operator:
        """Returns the Operator to be applied on qubits q, q+1 for a given time.

        Sadly, there are no parameterised operators which could be used to make them
        time-independent and avoid constructing a new circuit for each time step...
        TODO: find out if there is a representation for this operator in standard gates
        """
        return Operator([
            [np.exp(1j * time * (+self.h + self.J)), 0, 0, 0],
            [0, np.exp(1j * time * (+self.h - self.J)), 0, 0],
            [0, 0, np.exp(1j * time * (-self.h - self.J)), 0],
            [0, 0, 0, np.exp(1j * time * (-self.h + self.J))]
        ])

    def circuit(self, initial_state: QuantumCircuit, delta_t: float) -> QuantumCircuit:
        """Construct the quantum circuit to step the initial state by time delta_t.

        Sadly, this constructs a new circuit on each call for a new time.
        """
        if initial_state.num_qubits != self.N:
            raise ValueError(f"Instead of {self.N}, {initial_state.num_qubits} qubits"
                             f" were provided: {initial_state.qubits}.")
        operator = self._operator(delta_t)
        instruction: Instruction = operator.to_instruction()

        # apply the operator pairwise to all neighbour qubits.
        for bit in range(self.N):
            initial_state.append(instruction, [bit, (bit + 1) % self.N])

        return initial_state

    def simulate(self, initial_state: QuantumCircuit, stop_time: float,
                 accuracy: float = None,) -> QuantumCircuit:
        """Construct the circuit for a simulation with given boundary conditions.

        This can later be used to f.e. estimate the energy with phase estimation.
        """
        # Since all parts of the Hamiltonian commute, the unitary matrix provided in the
        # operator function is _exact_. Therefore, we always get the correct result with
        # arbitrary precision, and its the same if you call the circuit once for a given
        # time t or repeatedly for summands of t.
        return self.circuit(initial_state, delta_t=stop_time)


def random_angles(n_qubits: int, n_samples: int) -> List[List[Tuple[float, float]]]:
    """Draws a pair of angles from [0, 2*pi) for each Qubit in each sample."""
    return [
        [
            (np.random.uniform(low=0, high=2 * np.pi),
             np.random.uniform(low=0, high=2 * np.pi))
            for _ in range(n_qubits)
        ]
        for _ in range(n_samples)
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

    for bit, (sigma, theta) in enumerate(angles):
        # prepare an arbitrary state on the bloch sphere
        circ.rx(sigma, bit)
        circ.rz(theta, bit)

        # add CNOTs to accomplish entanglement between the qubits
        for prev_bit in range(bit):
            circ.cnot(control_qubit=bit, target_qubit=prev_bit)

    return circ


def init_cb_state(qubits: int, start_qubit: int) -> QuantumCircuit:
    """Returns a QuantumCircuit initialized in a Computational Basis state.

    This is, flipping the start_qubit to |1> and leave all others in |0>.
    """
    circ = QuantumCircuit(qubits)
    circ.x(start_qubit)
    return circ

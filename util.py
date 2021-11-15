import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, List, Set

import numpy as np
from time import time

from qiskit import QuantumCircuit
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.models.backendconfiguration import QasmBackendConfiguration
from qiskit.providers.aer.jobs.aerjob import AerJob
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


def init_random_state(qubits: int, angles: List[Tuple[float, float]]) -> QuantumCircuit:
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
    circ = QuantumCircuit(qubits)
    circ.x(start_qubit)
    return circ


model = LIModel(J=-1, h=2, N=4)
samples = 10    # how many initial states do we create to simulate
delta_t = 1     # time step for the ising model
ising_routine = model.circuit(QuantumCircuit(model.N), delta_t=delta_t)

backend: AerBackend = Aer.get_backend('aer_simulator')
backend.set_option("max_parallel_threads", 3)
phase_estimator = PhaseEstimation(num_evaluation_qubits=6, quantum_instance=backend)

phases: List[Dict[str, float]] = []

start = time()

for loop in range(samples):
    for bit in range(model.N):
        # the phase estimation energy caluclation is only true if the initial state is a
        # eigenstate of U. Since U is diagonal, the eigenstates are the computational
        # basis states.
        # TODO: this argument is given for the eigenstates of _H_, not for those of _U_!
        circ = init_cb_state(model.N, start_qubit=bit)
        circ.append(ising_routine.to_instruction(), [bit for bit in range(model.N)])

        # Transpile for simulator
        circ = transpile(circ, backend)
        # circ.draw(filename=f"model-round-{loop}-bit-{bit}.png", output="mpl")

        phase_circ = phase_estimator.construct_circuit(circ)
        phase = phase_estimator.estimate(circ)
        # the phase is computed in [0.0, 1.0)
        phases.append(phase.filter_phases(cutoff=0.001))
        print(f"Round {loop}, bit {bit}: {time() - start}")

pprint(phases)

exit()

samples_angles = [
    [
        # draw the angles (sigma, theta) for the (Rx, Rz) gate uniformly from [0, 2*pi)
        (np.random.uniform(low=0, high=2*np.pi), np.random.uniform(low=0, high=2*np.pi))
        for _ in range(model.N)
    ]
    for _ in range(samples)
]

for loop, angles in enumerate(samples_angles, start=1):
    circ = init_random_state(model.N, angles)
    circ.append(ising_routine.to_instruction(), [bit for bit in range(model.N)])

    # Transpile for simulator
    circ = transpile(circ, backend)
    circ.draw(filename=f"model-round-{loop}.png", output="mpl")

    phase_circ = phase_estimator.construct_circuit(circ)
    phase = phase_estimator.estimate(circ)
    # the phase is computed in [0.0, 1.0)
    phases.append(phase.filter_phases(cutoff=0.01, as_float=False))
    print(f"Round {loop}: {time() - start}")

pprint(phases)

all_phasestrings = [phase for phase_dict in phases for phase in phase_dict]
unique_phasestrings = set(all_phasestrings)
phase_count = {bitstring: all_phasestrings.count(bitstring) for bitstring in unique_phasestrings}
# sort dicts by value
phase_count = sorted(phase_count.items(), key=lambda e: e[1])
pprint(phase_count)

phase_sum = {bitstring: sum(mea.get(bitstring, 0) for mea in phases) for bitstring in unique_phasestrings}
# sort dicts by value
phase_sum = sorted(phase_sum.items(), key=lambda e: e[1])
pprint(phase_sum)

exit()

# Run and get counts
#result: Result = backend.run(circ, seed_simulator=3710243989).result()
job: AerJob = backend.run(circ)
# print(job.qobj())
result: Result = job.result()
experiment_result: List[ExperimentResult] = result.results
print(experiment_result[0].header)
counts = result.get_counts()
print(counts)
circ.draw(filename=f"limodel.png", output="mpl")

plt.show()  # to ensure a new figure is created
plot_histogram(counts, title="Longitudinal Ising Model")

plt.savefig("LI Model.png")

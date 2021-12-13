from qiskit import QuantumCircuit, AncillaRegister, QuantumRegister, ClassicalRegister
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

from w import OrderedOperator
from ordered_operator import IGate, ZGate


def hadamard_test_circ(state: QuantumCircuit, operator: OrderedOperator) -> QuantumCircuit:
    """Obtain Re(<state|operator|state>) using the Hadamard Test.

    The QM expectation value is the statistical expectation value of the measurement
    from the Hadamard test.

    Since we use only Z and I Gates as operators, its sufficient to look a the real part
    of the QM expectation value.

    Reference: https://en.wikipedia.org/wiki/Hadamard_test_(quantum_computation)
    """
    if state.num_qubits != operator.Q:
        raise ValueError("State and operator must have the same number ob qubits.")

    # create new registers for the ancilla qubit, the measure and the state |psi>.
    ancilla_reg = AncillaRegister(size=1)
    classic_reg = ClassicalRegister(size=1)
    state_reg = QuantumRegister(size=state.num_qubits)

    circ = QuantumCircuit(ancilla_reg, state_reg, classic_reg, name="Hadamard Test")

    # prepare the state on the new circ
    circ.append(state.to_instruction(), qargs=state_reg)
    circ.h(qubit=ancilla_reg)
    # apply the controlled operator to the state qubits. Since the operator consists
    # only of IGates and ZGates, this is rather easy.
    for gate in operator.gates:
        if isinstance(gate, IGate):
            circ.i(gate.qubit)
        elif isinstance(gate, ZGate):
            circ.cz(control_qubit=ancilla_reg, target_qubit=state_reg[gate.qubit])
        else:
            raise NotImplementedError(gate)
    circ.h(qubit=ancilla_reg)

    # measure the ancilla bit
    circ.barrier()
    circ.measure(ancilla_reg, classic_reg)

    return circ


def hadamard_test_result(result: Result) -> float:
    """Convert the measurement outcome of the hadamard test circ to the QM expectation value.

    We calculate the probability for measuring 0, then for measuring 1. The real part of
    the expectation value is then the difference between the two probabilities.
    """
    if len(result.results) != 1:
        raise RuntimeError("More than one experiment is present.")

    shots: int = result.results[0].shots
    counts = result.get_counts()
    count_0 = counts.get("0", 0)
    count_1 = counts.get("1", 0)

    if count_0 + count_1 != shots:
        raise ValueError(f"0_count {count_0} + 1_count {count_1} != shots {shots}.")

    prob_0 = count_0 / shots
    prob_1 = count_1 / shots

    # TODO absolute value?
    return prob_0 - prob_1

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


def my_test_circ(state: QuantumCircuit, operator: OrderedOperator) -> QuantumCircuit:
    """Obtain <state|operator|state> by literally applying the state and its dagger."""
    if state.num_qubits != operator.Q:
        raise ValueError("State and operator must have the same number ob qubits.")

    # create new registers
    classic_reg = ClassicalRegister(size=state.num_qubits)
    state_reg = QuantumRegister(size=state.num_qubits)

    circ = QuantumCircuit(state_reg, classic_reg, name="My Test")

    # apply <psi| ...
    circ.append(state.to_instruction(), qargs=state_reg)

    # ... apply operator ...
    for gate in operator.gates:
        if isinstance(gate, IGate):
            circ.i(state_reg[gate.qubit])
        elif isinstance(gate, ZGate):
            circ.z(state_reg[gate.qubit])
        else:
            raise NotImplementedError(gate)

    # ... apply |psi>
    circ.append(state.inverse().to_instruction(), qargs=state_reg)

    circ.barrier()
    circ.measure(state_reg, classic_reg)

    return circ


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

    # take care that the state-reg is before the ancilla-reg to map the noise properly!
    circ = QuantumCircuit(state_reg, ancilla_reg, classic_reg, name="Hadamard Test")

    # prepare the state on the new circ
    circ.append(state.to_instruction(), qargs=state_reg)
    circ.h(qubit=ancilla_reg)
    # apply the controlled operator to the state qubits. Since the operator consists
    # only of IGates and ZGates, this is rather easy.
    for gate in operator.gates:
        if isinstance(gate, IGate):
            circ.i(state_reg[gate.qubit])
        elif isinstance(gate, ZGate):
            circ.cz(control_qubit=ancilla_reg, target_qubit=state_reg[gate.qubit])
        else:
            raise NotImplementedError(gate)
    circ.h(qubit=ancilla_reg)

    # measure the ancilla bit
    circ.barrier()
    circ.measure(ancilla_reg, classic_reg)

    return circ


def my_test_result(result: Result, operator: OrderedOperator) -> float:
    """Convert the measurement outcome to the QM expectation value.

    Its assumed that each qubit is independently affected from the bitflip noise. So,
    we can calculate the expectation value for each bit separately:

    For only one bit, the expectation value reads:
        (count_0 - count_1) / shots

    For two bits, we sum up for each bit all 0 count and all 1 counts and divide by shots.
    So, for the first bit we get:
        (count_00 + count_01 - count_10 - count_11) / shots
    and for the second bit we get:
        (count_00 - count_01 + count_10 - count_11) / shots
    The overall result is obtained by multiplying the results for both bits.
    """
    if len(result.results) != 1:
        raise RuntimeError("More than one experiment is present.")

    shots: int = result.results[0].shots
    counts = result.get_counts()

    # validate the counts dict
    if any(len(key) != operator.Q for key in counts):
        raise ValueError("Malformed key in counts dict")
    if sum(counts.values()) != shots:
        raise ValueError("The entries of the counts dict doesn't sum up to the number of shots.")

    # all possible measurement outcomes and thereby all possible keys of the counts dict
    keys = [format(key, f'0{operator.Q}b') for key in range(2**operator.Q)]
    print("\t\t".join(f"{key}:{counts.get(key, 0)}" for key in keys), "\t\t", operator)

    expectation = 1
    for qubit in range(operator.Q):
        # determine all keys where the current qubit was measured as 1 / 0
        # normally we enumerate our qubits from right to left, however since we iterate
        # over all qubits, it doesn't matter here
        one_keys = [key for key in keys if key[qubit] == '1']
        zero_keys = [key for key in keys if key[qubit] == '0']
        expectation *= (sum(counts.get(key, 0) for key in zero_keys) - sum(counts.get(key, 0) for key in one_keys)) / shots

    return expectation


def hadamard_test_result(result: Result, operator=None) -> float:
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
    print("\t".join([str(value) for key, value in sorted(counts.items())]), "\t\t\t", operator)

    if count_0 + count_1 != shots:
        raise ValueError(f"0_count {count_0} + 1_count {count_1} != shots {shots}.")

    return (count_0 - count_1) / shots


def projective_measurement(result: Result, operator: OrderedOperator) -> float:
    if len(result.results) != 1:
        raise RuntimeError("More than one experiment is present.")

    shots: int = result.results[0].shots
    counts = result.get_counts()

    # validate the counts dict
    if any(len(key) != operator.Q for key in counts):
        raise ValueError("Malformed key in counts dict")
    if sum(counts.values()) != shots:
        raise ValueError("The entries of the counts dict doesn't sum up to the number of shots.")

    # all possible measurement outcomes and thereby all possible keys of the counts dict
    keys = [format(key, f'0{operator.Q}b') for key in range(2**operator.Q)]
    #print("\t\t".join(f"{key}:{operator.sign(key)}{counts.get(key, 0)}" for key in keys), "\t\t", operator)

    return sum(operator.sign(key) * counts[key] for key in counts) / shots

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile

from w import ErrorProbabilities, p0_symbol, p1_symbol


def measure_errors(backend, N: int) -> ErrorProbabilities:
    """Measure the error probabilities for the given number of qubits."""
    ret = dict()
    shots = backend.options.get("shots")
    for qubit in range(N):
        qreg = QuantumRegister(N)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.measure(qubit, creg)
        circ = transpile(circ, backend, seed_transpiler=42)
        counts = backend.run(circ).result().get_counts()
        ret[p0_symbol(qubit)] = counts.get("1", 0) / shots
    for qubit in range(N):
        qreg = QuantumRegister(N)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.x(qubit)
        circ.measure(qubit, creg)
        circ = transpile(circ, backend, seed_transpiler=42)
        counts = backend.run(circ).result().get_counts()
        ret[p1_symbol(qubit)] = counts.get("0", 0) / shots
    return ret

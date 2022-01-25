from typing import Dict
from sympy import Symbol

from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister

from w import p0_symbol, p1_symbol

# Mapping a symbol retrieved from w.py p0_symbol / p1_symbol to the flipping probability
ErrorProbabilities = Dict[Symbol, float]


def measure_errors(backend, qubits: int) -> ErrorProbabilities:
    """Measure the error probabilites of a given backend for the given number of qubits."""
    ret = dict()
    shots = backend.options.get("shots")
    for qubit in range(qubits):
        qreg = QuantumRegister(qubits)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.measure(qubit, creg)
        circ = transpile(circ, backend, seed_transpiler=42)
        counts = backend.run(circ).result().get_counts()
        ret[p0_symbol(qubit)] = counts.get("1", 0) / shots
    for qubit in range(qubits):
        qreg = QuantumRegister(qubits)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.x(qubit)
        circ.measure(qubit, creg)
        circ = transpile(circ, backend, seed_transpiler=42)
        counts = backend.run(circ).result().get_counts()
        ret[p1_symbol(qubit)] = counts.get("0", 0) / shots
    return ret

from typing import Dict, Union, overload, Sequence
from sympy import Symbol, zeros, Matrix, ImmutableMatrix, Expr
from itertools import product

from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import IGate, ZGate

# each gate which can be sorted in lexicographic order
OrderedGate = Union[IGate, ZGate]
# a operator acting on one or more qubits, containing an OrderedGate for each qubit
OrderedOperator = Sequence[OrderedGate]
# a bunch of operators acting on one or more qubits
OrderedOperators = Sequence[OrderedOperator]

# use global variables to cache calculated matrices
_W_MATRIX_DIMENSION: Dict[int, ImmutableMatrix] = dict()
_W_MATRIX_INVERSE_DIMENSION: Dict[int, ImmutableMatrix] = dict()


@overload
def lexicographic_less(left: OrderedGate, right: OrderedGate, equal: bool = False) -> bool:
    ...


@overload
def lexicographic_less(left: OrderedOperator, right: OrderedOperator, equal: bool = False) -> bool:
    ...


def lexicographic_less(left, right, equal: bool = False) -> bool:
    if not isinstance(left, Sequence):
        left = [left]
    if not isinstance(right, Sequence):
        right = [right]

    if not all(isinstance(gate, IGate) or isinstance(gate, ZGate) for gate in left):
        raise NotImplementedError("Comparison is not implemented for this gates.")
    if not all(isinstance(gate, IGate) or isinstance(gate, ZGate) for gate in right):
        raise NotImplementedError("Comparison is not implemented for this operator.")

    if len(left) != len(right):
        raise ValueError("Both operator must contain the same number of gates!")

    # we start comparing the "left" and "right" operator from the left pairwise
    for l, r in zip(left, right):
        # we can not say if on is larger than the other
        if isinstance(l, IGate) and isinstance(r, IGate):
            continue
        # we can not say if on is larger than the other
        elif isinstance(l, ZGate) and isinstance(r, ZGate):
            continue
        elif isinstance(l, IGate) and isinstance(r, ZGate):
            return True
        elif isinstance(l, ZGate) and isinstance(r, IGate):
            return False
        else:
            raise RuntimeError
    # the operators are equal
    return equal


def lexicographic_ordered_operators(Q: int) -> OrderedOperators:
    """Returns all combinations of Q operators, in lexicographic order."""
    # TODO: check if this works as expected
    return list(product([IGate(), ZGate()], repeat=Q))


def operator_position(operator: OrderedOperator) -> int:
    """The zero-based position of the operator in the vector of all lexicographic ordered operators of dimension Q.

    For example, the operator O=IZ has position 1, since the vector of all operators
    with dimension 2 in lexicographic order is (II, IZ, ZI, ZZ).

    Determine the order by casting the operator-tuple into a bitstring, replacing
    I -> 0 and Z -> 1.
    """
    bitstring = "".join(["1" if isinstance(gate, ZGate) else "0" for gate in operator])
    return int(bitstring, base=2)


def w_matrix(Q: int) -> ImmutableMatrix:
    """Return the w-matrix for a given number Q of operators.

    Suppose we have a total of Q operators (Pauli-Z or Identity), acting on Q qubits.
    Then, we consider the vector which entries are all possible combinations (with
    respect to the tensor product) of those operators of length Q. The entries of the
    operator are ordered by the _lexicographic order_ from top to bottom.

    Then, w is the matrix relating the noise-free operators O to the statistical
    expectation value of the operators \tilde{O}.

    For Q=2, this looks as follows (w is a 4x4 matrix, in general a 2^Qx2^Q matrix):
        ––                         ––            ––   ––
        | E ( \tilde{I} \tilde{I} ) |            | I I |
        | E ( \tilde{I} \tilde{Z} ) |            | I Z |
        | E ( \tilde{Z} \tilde{I} ) |   =  w  *  | Z I |
        | E ( \tilde{Z} \tilde{Z} ) |            | Z Z |
        ––                         ––            ––   ––

    The matrix is filled with sympy symbols for the bit-flip probabilities of each
    qubit, which should be inserted using sympy.substitute later one. In this way,
    the matrix for a number of Q qubits need to be computed only once.
    """
    # use cached matrix if available
    if Q in _W_MATRIX_DIMENSION:
        return _W_MATRIX_DIMENSION[Q]

    noiseless_operators = lexicographic_ordered_operators(Q)
    noisy_operators = lexicographic_ordered_operators(Q)

    matrix: Matrix = zeros(2**Q)

    # fill the matrix with content
    for row, noisy_operator in enumerate(noisy_operators):
        for col, noiseless_operator in enumerate(noiseless_operators):
            matrix[row, col] = w_element(noiseless_operator, noisy_operator)

    # cache matrix
    _W_MATRIX_DIMENSION[Q] = ImmutableMatrix(matrix)

    return _W_MATRIX_DIMENSION[Q]


def w_matrix_inverse(Q: int) -> ImmutableMatrix:
    """Calculate the inverse w matrix.

    This is used to actually error correct an operator from several noisy operator
    expectation values. Note that this still contains the placeholders for the
    probabilities.
    """
    # use cached matrix if available
    if Q in _W_MATRIX_INVERSE_DIMENSION:
        return _W_MATRIX_INVERSE_DIMENSION[Q]

    matrix = w_matrix(Q)
    _W_MATRIX_INVERSE_DIMENSION[Q] = matrix.inverse()

    return _W_MATRIX_INVERSE_DIMENSION[Q]


def w_element(noiseless_operator: OrderedOperator, noisy_operator: OrderedOperator):
    """Calculate one entry of the w_matrix.

    The entry depends on the noise free operators (column) and noisy operators (row).

    For Q=1, the 2x2 w-matrix looks like this:
        ––                                                 ––
        | w_element(I, \tilde{I})   w_element(Z, \tilde{I}) |
        | w_element(I, \tilde{Z})   w_element(Z, \tilde{Z}) |
        ––                                                 ––
    """
    # w is a lower triangular matrix, so the upper right part of the matrix is 0
    if lexicographic_less(noisy_operator, noiseless_operator):
        return 0

    ret = 1
    for qubit, (noiseless_gate, noisy_gate) in enumerate(zip(noiseless_operator, noisy_operator)):
        if isinstance(noiseless_gate, IGate) and isinstance(noisy_gate, IGate):
            ret *= 1
        elif isinstance(noiseless_gate, IGate) and isinstance(noisy_gate, ZGate):
            p0_q = p0_symbol(qubit)
            p1_q = p1_symbol(qubit)
            ret *= p1_q - p0_q
        elif isinstance(noiseless_gate, ZGate) and isinstance(noisy_gate, IGate):
            ret *= 0
            break
        elif isinstance(noiseless_gate, ZGate) and isinstance(noisy_gate, ZGate):
            p0_q = p0_symbol(qubit)
            p1_q = p1_symbol(qubit)
            ret *= 1 - p0_q - p1_q
        else:
            print(noiseless_operator)
            print(noisy_operator)
            raise ValueError(qubit, noiseless_gate, noisy_gate)

    return ret


def w_inverse_element(noiseless_operator: OrderedOperator, noisy_operator: OrderedOperator) -> Expr:
    """Get the entry in w^-1 for given noiseless and noisy operator.

    This is needed to reconstruct the noiseless operator from multiple noisy one.
    """
    if len(noiseless_operator) != len(noisy_operator):
        raise ValueError("Noisless and noisy operator must contain the same number of gates.")
    Q = len(noiseless_operator)
    row = operator_position(noiseless_operator)
    col = operator_position(noisy_operator)
    w_inverse = w_matrix_inverse(Q)
    return w_inverse[row, col]


def relevant_operators(noiseless_operator: OrderedOperator) -> OrderedOperators:
    """Determine all noisy operators which need to be measured to reconstruct the given noiseless operator.

    Go through all columns of w^-1 for the row given by the noiseless_operator and finde
    those matrix elements which are not 0.
    """
    Q = len(noiseless_operator)
    row = operator_position(noiseless_operator)
    noisy_operators = lexicographic_ordered_operators(Q)
    w_inverse = w_matrix_inverse(Q)
    return [operator for operator, w_entry in zip(noisy_operators, w_inverse.row(row)) if w_entry != 0]


def p0_symbol(q: int) -> Symbol:
    """Symbol for the bit-flip probability from |0> -> |1> of qubit q.

    We use sympy symbols to add placeholders into the w matrix. This placeholders will
    be substituted later using sympy.substitute, so we need to calculate the w matrix
    only once per number of operators Q.

    This functions is used to provide a consistent naming scheme.
    """
    return Symbol(f"p0_{q}")


def p1_symbol(q: int) -> Symbol:
    """Symbol for the bit-flip probability from |1> -> |0> of qubit q.

    We use sympy symbols to add placeholders into the w matrix. This placeholders will
    be substituted later using sympy.substitute, so we need to calculate the w matrix
    only once per number of operators Q.

    This functions is used to provide a consistent naming scheme.
    """
    return Symbol(f"p1_{q}")


def operator_to_circ(operator: OrderedOperator) -> QuantumCircuit:
    qubits = len(operator)
    circ = QuantumCircuit(qubits)
    for qubit, gate in enumerate(operator):
        if isinstance(gate, IGate):
            circ.i(qubit)
        elif isinstance(gate, ZGate):
            circ.z(qubit)
        else:
            raise RuntimeError
    return circ

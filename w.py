from typing import List, Union, overload, Sequence
from sympy import Symbol, zeros, Matrix, ImmutableMatrix
from itertools import product

from qiskit.circuit.library.standard_gates import IGate, ZGate

OrderedGate = Union[IGate, ZGate]
OrderedGates = Sequence[OrderedGate]


@overload
def lexicographic_less(left: OrderedGate, right: OrderedGate, equal: bool = False) -> bool:
    ...


@overload
def lexicographic_less(left: OrderedGates, right: OrderedGates, equal: bool = False) -> bool:
    ...


def lexicographic_less(left, right, equal: bool = False) -> bool:
    if not isinstance(left, Sequence):
        left = [left]
    if not isinstance(right, Sequence):
        right = [right]

    if not all(isinstance(gate, IGate) or isinstance(gate, ZGate) for gate in left):
        raise NotImplementedError("Comparison is not implemented for this gates.")
    if not all(isinstance(gate, IGate) or isinstance(gate, ZGate) for gate in right):
        raise NotImplementedError("Comparison is not implemented for this gates.")

    if len(left) != len(right):
        raise ValueError("Both arguments must have the same length!")

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


def lexicographic_ordered_operators(Q: int) -> List[OrderedGates]:
    """Returns all combinations of Q operators, in lexicographic order."""
    # TODO: check if this works as expected
    return list(product([IGate(), ZGate()], repeat=Q))


def w_matrix(Q: int) -> ImmutableMatrix:
    """Return the w-matrix for a given number Q of operators.

    Suppose we have a total of Q operators (Pauli-Z or Identity), acting on Q qubits.
    Then, we consider the vector which entries are all possible combinations (with
    respect to the tensor product) of those operators of length Q. The entries of the
    operator are ordered by the _lexicographic order_ from top to bottom.

    Then, w is the matrix relating the noise-free operators O to the statistical
    expectation value of the operators \tilde{O}.

    For Q=2, this looks as follows (w is a 4x4 matrix):
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
    noiseless_operators = lexicographic_ordered_operators(Q)
    noisy_operators = lexicographic_ordered_operators(Q)

    matrix: Matrix = zeros(Q)

    # fill the matrix with content
    for row, noisy_operator in enumerate(noisy_operators):
        for col, noiseless_operator in enumerate(noiseless_operators):
            matrix[row, col] = w_element(noiseless_operator, noisy_operator)

    return ImmutableMatrix(matrix)


def w_element(noiseless_operators: OrderedGates, noisy_operators: OrderedGates):
    """Calculate one entry of the w_matrix.

    The entry depends on the noise free operators (column) and noisy operators (row).

    For Q=1, the 2x2 w-matrix looks like this:
        ––                                                 ––
        | w_element(I, \tilde{I})   w_element(Z, \tilde{I}) |
        | w_element(I, \tilde{Z})   w_element(Z, \tilde{Z}) |
        ––                                                 ––
    """
    # w is a lower triangular matrix, so the upper right part of the matrix is 0
    if lexicographic_less(noisy_operators, noiseless_operators):
        return 0

    ret = 1
    for qubit, (noiseless_operator, noisy_operator) in enumerate(zip(noiseless_operators, noisy_operators)):
        if isinstance(noiseless_operator, IGate) and isinstance(noisy_operator, IGate):
            ret *= 1
        elif isinstance(noiseless_operator, IGate) and isinstance(noisy_operator, ZGate):
            p0_q = p0_symbol(qubit)
            p1_q = p1_symbol(qubit)
            ret *= p1_q - p0_q
        # this can not happen, since this is only the case in the upper triangular
        # part of the matrix, which is caught separately above.
        # elif isinstance(noiseless_operator, ZGate) and isinstance(noisy_operator, IGate):
        elif isinstance(noiseless_operator, ZGate) and isinstance(noisy_operator, ZGate):
            p0_q = p0_symbol(qubit)
            p1_q = p1_symbol(qubit)
            ret *= 1 - p0_q - p1_q
        else:
            raise ValueError

    return ret


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

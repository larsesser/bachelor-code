import re

from typing import Dict, NamedTuple
from sympy import Symbol, zeros, Matrix, ImmutableMatrix, Expr, S
from sympy.printing import sstr

from ordered_operator import IGate, ZGate, OrderedOperator, OrderedOperators, lexicographic_ordered_operators

# use global variables to cache calculated matrices
_W_MATRIX_DIMENSION: Dict[int, ImmutableMatrix] = dict()
_W_MATRIX_INVERSE_DIMENSION: Dict[int, ImmutableMatrix] = dict()


def w_matrix(Q: int, error_probabilities) -> ImmutableMatrix:
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

    To save a great amount of computation time when we later need to invert the matrix,
    the error probabilities are directly substituted with the given values.
    """
    # use cached matrix if available
    if Q in _W_MATRIX_DIMENSION:
        return _W_MATRIX_DIMENSION[Q]

    # check given error probabilities
    if not all(unravel_symbol(symbol).qubit in range(Q) for symbol in error_probabilities.keys()):
        raise ValueError("A given error affects a qubit which is out of range of the given operator.")
    if 2 * Q != len(error_probabilities):
        raise ValueError("Not all error probabilities were specified"
                         " (flipping |0> -> |1> and flipping |1> -> |0>.")
    # ensure the probabilities are taken accurate as sympy numbers
    error_probabilities = {key: S(value) for key, value in error_probabilities.items()}

    noiseless_operators = lexicographic_ordered_operators(Q)
    noisy_operators = lexicographic_ordered_operators(Q)

    matrix: Matrix = zeros(2**Q)

    # fill the matrix with content
    for row, noisy_operator in enumerate(noisy_operators):
        for col, noiseless_operator in enumerate(noiseless_operators):
            matrix[row, col] = w_element(noiseless_operator, noisy_operator)

    # substitute the probabilities in the matrix, cut-off very small numbers (<< 10^-50)
    matrix = matrix.evalf(subs=error_probabilities, chop=True)

    # cache matrix
    _W_MATRIX_DIMENSION[Q] = ImmutableMatrix(matrix)

    return _W_MATRIX_DIMENSION[Q]


def w_matrix_inverse(Q: int) -> ImmutableMatrix:
    """Calculate the inverse w matrix.

    This is used to actually error correct an operator from several noisy operator
    expectation values. Note that the values for the probabilities are directly inserted.
    """
    # use cached matrix if available
    if Q in _W_MATRIX_INVERSE_DIMENSION:
        return _W_MATRIX_INVERSE_DIMENSION[Q]

    if Q not in _W_MATRIX_DIMENSION:
        raise RuntimeError("First call w_matrix explicitly to compute the matrix.")
    matrix = _W_MATRIX_DIMENSION[Q]

    _W_MATRIX_INVERSE_DIMENSION[Q] = matrix.inverse()

    return _W_MATRIX_INVERSE_DIMENSION[Q]


def w_element(noiseless_operator: OrderedOperator, noisy_operator: OrderedOperator) -> Expr:
    """Calculate one entry of the w_matrix.

    The entry depends on the noise free operators (column) and noisy operators (row).

    For Q=1, the 2x2 w-matrix looks like this:
        ––                                                 ––
        | w_element(I, \tilde{I})   w_element(Z, \tilde{I}) |
        | w_element(I, \tilde{Z})   w_element(Z, \tilde{Z}) |
        ––                                                 ––
    """
    # w is a lower triangular matrix, so the upper right part of the matrix is 0
    if noisy_operator < noiseless_operator:
        return S(0)

    ret = S(1)
    for noiseless_gate, noisy_gate in zip(noiseless_operator.gates, noisy_operator.gates):
        if noisy_gate.qubit != noiseless_gate.qubit:
            raise ValueError("NoisyGate and NoislessGate must act on the same qubit!")
        qubit = noisy_gate.qubit
        if isinstance(noiseless_gate, IGate) and isinstance(noisy_gate, IGate):
            ret *= S(1)
        elif isinstance(noiseless_gate, IGate) and isinstance(noisy_gate, ZGate):
            p0_q = p0_symbol(qubit)
            p1_q = p1_symbol(qubit)
            ret *= (p1_q - p0_q)
        elif isinstance(noiseless_gate, ZGate) and isinstance(noisy_gate, IGate):
            ret *= S(0)
            break
        elif isinstance(noiseless_gate, ZGate) and isinstance(noisy_gate, ZGate):
            p0_q = p0_symbol(qubit)
            p1_q = p1_symbol(qubit)
            ret *= (S(1) - p0_q - p1_q)
        else:
            print(noiseless_operator)
            print(noisy_operator)
            raise ValueError(qubit, noiseless_gate, noisy_gate)

    return ret


def w_inverse_element(noiseless_operator: OrderedOperator, noisy_operator: OrderedOperator) -> Expr:
    """Get the entry in w^-1 for given noiseless and noisy operator.

    This is needed to reconstruct the noiseless operator from multiple noisy one.
    """
    if noiseless_operator.Q != noisy_operator.Q:
        raise ValueError("Noisless and noisy operator must contain the same number of gates.")
    row = noiseless_operator.position
    col = noisy_operator.position
    w_inverse = w_matrix_inverse(noiseless_operator.Q)
    return w_inverse[row, col]


def relevant_operators(noiseless_operator: OrderedOperator) -> OrderedOperators:
    """Determine all noisy operators which need to be measured to reconstruct the given noiseless operator.

    Go through all columns of w^-1 for the row given by the noiseless_operator and finde
    those matrix elements which are not 0.
    """
    row = noiseless_operator.position
    noisy_operators = lexicographic_ordered_operators(noiseless_operator.Q)
    w_inverse = w_matrix_inverse(noiseless_operator.Q)
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


class Bitflip(NamedTuple):
    from_state: int
    to_state: int
    qubit: int


def unravel_symbol(symbol: Symbol) -> Bitflip:
    """Extract the information from a bitflip-probability symbol.

    This takes one of the symbols created by p0_symbol or p1_symbol and extracts the
    flipping (from_state flipping into to_state) and the affected qubit.
    """
    symbol_str = sstr(symbol)
    if m := re.match(r"p(?P<from_state>\d)_(?P<qubit>\d)", symbol_str):
        if m.group("from_state") not in {"0", "1"}:
            raise ValueError("Not a from_state.")
        from_state = int(m.group("from_state"))
        to_state = 0 if from_state == 1 else 1
        qubit = int(m.group("qubit"))
        return Bitflip(from_state=from_state, to_state=to_state, qubit=qubit)
    else:
        raise ValueError("Not a matching symbol.")

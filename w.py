import re
from typing import Dict, NamedTuple

from sympy import Expr, ImmutableMatrix, Matrix, S, Symbol, zeros
from sympy.printing import sstr

from ordered_operator import (IGate, OrderedOperator, ZGate,
                              lexicographic_ordered_operators)

# use global variables to cache calculated matrices
_W_MATRIX: Dict[int, ImmutableMatrix] = dict()
_W_MATRIX_INVERSE: Dict[int, ImmutableMatrix] = dict()

# Mapping a symbol to the flipping probability
ErrorProbabilities = Dict[Symbol, float]


def w_matrix(N: int, errors: ErrorProbabilities) -> ImmutableMatrix:
    """Return the w-matrix for a given operator dimension N.

    All noiseless operators of dimension N consist of N gates (Pauli-Z or
    Identity). Suppose they are ordered in a vector \vec{O} with respect to the
    lexicographic order. Similar, all noisy operators of dimension N are
    ordered in a vector \vec{\tilde{O}}, also with respect to the lexicographic
    order.

    Then, w is the matrix relating the noiseless operators O of \vec{O} to the
    noisy operators \tilde{O} of \vec{\tilde{O}}.

    For N=2, this looks as follows (w is a 2^Nx2^N matrix):
        ––                   ––            ––   ––
        | \tilde{I} \tilde{I} |            | I I |
        | \tilde{I} \tilde{Z} |            | I Z |
        | \tilde{Z} \tilde{I} |   =  w  *  | Z I |
        | \tilde{Z} \tilde{Z} |            | Z Z |
        ––                   ––            ––   ––

    To save a great amount of computation time when we later need to invert the
    matrix, the placeholders of the error probabilities are substituted with
    the given values.
    """
    # use cached matrix if available
    if N in _W_MATRIX:
        return _W_MATRIX[N]

    # check given error probabilities
    if any(unravel_symbol(symbol).qubit not in range(N) for symbol in errors):
        raise ValueError("An error affects a qubit outside the given range.")
    if 2 * N != len(errors):
        raise ValueError("Not all error probabilities were specified"
                         " (flipping |0> -> |1> and flipping |1> -> |0>).")
    # ensure the probabilities are taken accurate as sympy numbers
    errors = {key: S(value) for key, value in errors.items()}

    noiseless_operators = lexicographic_ordered_operators(N)
    noisy_operators = lexicographic_ordered_operators(N)

    matrix: Matrix = zeros(2**N)

    # fill the matrix with content
    for row, noisy_operator in enumerate(noisy_operators):
        for col, noiseless_operator in enumerate(noiseless_operators):
            matrix[row, col] = w_element(noiseless_operator, noisy_operator)

    # substitute the probabilities, cut-off very small numbers (<< 10^-50)
    matrix = matrix.evalf(subs=errors, chop=True)

    # cache matrix
    _W_MATRIX[N] = ImmutableMatrix(matrix)

    return _W_MATRIX[N]


def w_matrix_inverse(N: int) -> ImmutableMatrix:
    """Calculate the inverse w matrix.

    This is used to actually error correct an operator from several noisy
    operator expectation values. Note that the values for the probabilities
    are already inserted.
    """
    # use cached matrix if available
    if N in _W_MATRIX_INVERSE:
        return _W_MATRIX_INVERSE[N]

    if N not in _W_MATRIX:
        raise RuntimeError("First, call w_matrix explicitly.")
    matrix = _W_MATRIX[N]

    _W_MATRIX_INVERSE[N] = matrix.inverse()

    return _W_MATRIX_INVERSE[N]


def w_element(noiseless_operator: OrderedOperator,
              noisy_operator: OrderedOperator) -> Expr:
    """Calculate one entry of the w_matrix.

    The entry depends on the noiseless operator (determining the column) and
    the noisy operator (determining the row).

    For N=1, the 2x2 w-matrix looks like this:
        ––                                                 ––
        | w_element(I, \tilde{I})   w_element(Z, \tilde{I}) |
        | w_element(I, \tilde{Z})   w_element(Z, \tilde{Z}) |
        ––                                                 ––
    """
    # w is a lower triangular matrix, so the upper right part is always 0
    if noisy_operator < noiseless_operator:
        return S(0)

    ret = S(1)
    for noiseless_gate, noisy_gate in zip(noiseless_operator.gates,
                                          noisy_operator.gates):
        if noisy_gate.qubit != noiseless_gate.qubit:
            raise ValueError("Both gates must act on the same qubit!")
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
            raise NotImplementedError

    return ret


def w_inverse_element(noiseless_operator: OrderedOperator,
                      noisy_operator: OrderedOperator) -> Expr:
    """Get the entry in w^-1 for the given noiseless and noisy operator.

    This is used to reconstruct the noiseless operator from multiple noisy ones.
    """
    if noiseless_operator.N != noisy_operator.N:
        raise ValueError("Both operators must have the same dimension.")
    row = noiseless_operator.position
    col = noisy_operator.position
    w_inverse = w_matrix_inverse(noiseless_operator.N)
    return w_inverse[row, col]


def p0_symbol(q: int) -> Symbol:
    """Symbol for the bit-flip probability from |0> -> |1> of qubit q.

    We use sympy symbols to add placeholders into the w matrix to avoid machine
    precision errors and make the code more readable. This function is used to
    provide a consistent naming scheme.
    """
    return Symbol(f"p0_{q}")


def p1_symbol(q: int) -> Symbol:
    """Symbol for the bit-flip probability from |1> -> |0> of qubit q.

    We use sympy symbols to add placeholders into the w matrix to avoid machine
    precision errors and make the code more readable. This function is used to
    provide a consistent naming scheme.
    """
    return Symbol(f"p1_{q}")


class Bitflip(NamedTuple):
    from_state: int
    to_state: int
    qubit: int


def unravel_symbol(symbol: Symbol) -> Bitflip:
    """Extract the information from a bitflip-probability symbol.

    This takes one of the symbols created by p0_symbol or p1_symbol and
    extracts the flipping direction (from_state flipping into to_state) and the
    affected qubit.
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

from functools import total_ordering
from typing import List, Sequence, Tuple
from itertools import product

# a bunch of operators acting on one or more qubits
OrderedOperators = Sequence["OrderedOperator"]


@total_ordering
class OrderedGate:
    # the name of the operator
    name: str
    # the order of the operator, considered by the lexicographic order.
    order: int = 0
    # used to determine the position of the operator containing this gate
    as_bitstring: str

    def __eq__(self, other):
        if not isinstance(other, OrderedGate):
            raise NotImplementedError("Can only compare OrderedGates!")
        return self.order == other.order

    def __lt__(self, other):
        if not isinstance(other, OrderedGate):
            raise NotImplementedError("Can only compare OrderedGates!")
        return self.order < other.order


class IGate(OrderedGate):
    name = "I"
    order = 1
    as_bitstring = "0"

    def __str__(self):
        return "I"

    def __repr__(self):
        return self.__str__()


class ZGate(OrderedGate):
    name = "Z"
    order = 2
    as_bitstring = "1"

    def __str__(self):
        return "Z"

    def __repr__(self):
        return self.__str__()


@total_ordering
class OrderedOperator:
    gates: Tuple[OrderedGate]

    def __init__(self, *gates: OrderedGate):
        if not all(isinstance(gate, OrderedGate) for gate in gates):
            raise ValueError("Expected a sequence of OrderedGates!")
        self.gates = tuple(gates)

    def __eq__(self, other):
        if not isinstance(other, OrderedOperator):
            raise NotImplementedError("Can only compare OrderedOperators!")
        if len(self.gates) != len(other.gates):
            raise ValueError("Both operators must have the same number of gates!")
        return self.gates == other.gates

    def __lt__(self, other):
        if not isinstance(other, OrderedOperator):
            raise NotImplementedError("Can only compare OrderedOperators!")
        if len(self.gates) != len(other.gates):
            raise ValueError("Both operators must have the same number of gates!")
        return self.gates < other.gates

    def __str__(self):
        return "[" + ", ".join(str(gate) for gate in self.gates) + "]"

    def __repr__(self):
        return self.__str__()

    @property
    def position(self) -> int:
        """The zero-based position of the operator in the vector of all lexicographic ordered operators of same dimension.

        For example, the operator O=IZ has position 1, since the vector of all operators
        with dimension 2 in lexicographic order is (II, IZ, ZI, ZZ).

        Determine the order by casting the operator-tuple into a bitstring, replacing
        I -> 0 and Z -> 1.
        """
        bitstring = "".join([gate.as_bitstring for gate in self.gates])
        return int(bitstring, base=2)

    @property
    def Q(self) -> int:
        """The dimension of the operator."""
        return len(self.gates)


def lexicographic_ordered_operators(Q: int) -> OrderedOperators:
    """Returns all combinations of Q operators, in lexicographic order."""
    # TODO: check if this works as expected
    operators: List[Tuple[OrderedGate, ...]] = list(product([IGate(), ZGate()], repeat=Q))
    return [OrderedOperator(*operator) for operator in operators]

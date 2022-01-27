from functools import total_ordering
from itertools import product
from typing import List, Sequence, Tuple

# a bunch of operators acting on one or more qubits
OrderedOperators = Sequence["OrderedOperator"]


@total_ordering
class OrderedGate:
    """A single gate acting on one qubit and supporting lexicographic order."""
    # the name of the operator
    name: str
    # the qubit on which the gate will act
    qubit: int
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

    def __init__(self, qubit: int):
        self.qubit = qubit


class IGate(OrderedGate):
    name = "I"
    order = 1
    as_bitstring = "0"

    def __str__(self):
        return f"I({self.qubit})"

    def __repr__(self):
        return self.__str__()


class ZGate(OrderedGate):
    name = "Z"
    order = 2
    as_bitstring = "1"

    def __str__(self):
        return f"Z({self.qubit})"

    def __repr__(self):
        return self.__str__()


@total_ordering
class OrderedOperator:
    """An operator consisting of multiple gates which support lexicographic order."""
    gates: Tuple[OrderedGate]

    def __init__(self, *gates: OrderedGate):
        if not all(isinstance(gate, OrderedGate) for gate in gates):
            raise ValueError("Expected a sequence of OrderedGates!")
        qubits = [gate.qubit for gate in gates]
        if len(set(qubits)) != len(qubits):
            raise ValueError("All gates must act on differen qubits!")
        qubits2 = qubits.copy()
        qubits2.sort(reverse=True)
        if qubits != qubits2:
            raise ValueError("Gates must be provided in descending order!")
        if len(qubits) != (qubits[0] + 1):
            raise ValueError("A gate must be applied to all qubits inbetween!")
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
    def N(self) -> int:
        """The dimension of the operator.

        This is determined by the number of gates of the operator and therefore equal to
        the number of qubits the operator acts on.
        """
        return len(self.gates)

    def sign(self, bitstring: str) -> int:
        """The sign of the given bitstring w.r.t this operator to calculate the expectation value.

        The sign gets a factor of -1 for each 1 in the bitstring if the corresponding
        gate of the operator is a Z gate.
        """
        gate_string = "".join([gate.as_bitstring for gate in self.gates])
        one_and_z = bin(int(bitstring, base=2) & int(gate_string, base=2)).count("1")
        return (-1) ** one_and_z


def lexicographic_ordered_operators(N: int) -> OrderedOperators:
    """Returns all combinations of N dimensional operators, in lexicographic order."""
    # create all combinations of OrderedGates in lexicographic order
    # note that the qubit is only set temporarily, we will adjust it later
    prototypes: List[Tuple[OrderedGate, ...]] = list(product([IGate(-1), ZGate(-2)], repeat=N))

    operators: List[OrderedOperator] = list()
    for prototype in prototypes:
        # take care, since we enumerate our qubits conventionally from right to left!
        qubits = range(N)[::-1]
        gates: List[OrderedGate] = list()
        # adjust the qubit of each gate to the correct value
        # attention: all gates are the same instance, so we re-instantiate each gate!
        for qubit, gate in zip(qubits, prototype):
            if isinstance(gate, IGate):
                gate = IGate(qubit)
            elif isinstance(gate, ZGate):
                gate = ZGate(qubit)
            else:
                raise NotImplementedError
            gates.append(gate)
        operators.append(OrderedOperator(*gates))
    return operators

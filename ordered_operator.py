import itertools
from functools import total_ordering
from typing import List, Sequence, Tuple

# a bunch of operators acting on one or more qubits
OrderedOperators = Sequence["OrderedOperator"]


@total_ordering
class OrderedGate:
    """A single gate acting on one qubit and supporting lexicographic order."""
    # the name of the gate
    name: str
    # the gate acts on this qubit
    qubit: int
    # the order of the gate, considered by the lexicographic order.
    order: int = 0
    # express the gate as bit for advanced calculations in OrdereOperator
    as_bitstring: str

    def __eq__(self, other: "OrderedGate"):
        if not isinstance(other, OrderedGate):
            raise NotImplementedError("Can only compare OrderedGates!")
        return self.order == other.order

    def __lt__(self, other: "OrderedGate"):
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
    """Operator consisting of multiple gates supporting lexicographic order."""
    gates: Tuple[OrderedGate]

    def __init__(self, *gates: OrderedGate):
        if not all(isinstance(gate, OrderedGate) for gate in gates):
            raise ValueError("Expected a sequence of OrderedGates!")
        qubits = [gate.qubit for gate in gates]
        if len(set(qubits)) != len(qubits):
            raise ValueError("All gates must act on different qubits!")
        qubits2 = qubits.copy()
        qubits2.sort(reverse=True)
        if qubits != qubits2:
            raise ValueError("Gates must be provided in descending order!")
        if len(qubits) != (qubits[0] + 1):
            raise ValueError("A gate must be applied to all qubits inbetween!")
        self.gates = tuple(gates)

    def __eq__(self, other: "OrderedOperator"):
        if not isinstance(other, OrderedOperator):
            raise NotImplementedError("Can only compare OrderedOperators!")
        if len(self.gates) != len(other.gates):
            raise ValueError("Operators have different number of gates!")
        return self.gates == other.gates

    def __lt__(self, other: "OrderedOperator"):
        if not isinstance(other, OrderedOperator):
            raise NotImplementedError("Can only compare OrderedOperators!")
        if len(self.gates) != len(other.gates):
            raise ValueError("Operators have different number of gates!")
        return self.gates < other.gates

    def __str__(self):
        return "[" + ", ".join(str(gate) for gate in self.gates) + "]"

    def __repr__(self):
        return self.__str__()

    @property
    def position(self) -> int:
        """Position in the vector of all operators with equal dimension.

        The vector is sorted with respect to the lexicographic order from top
        to bottom. The position is zero-based.

        For example, the operator O=IZ has position 1, since the vector of all
        operators with dimension 2 in lexicographic order is (II, IZ, ZI, ZZ).

        Determine the order by casting the operator-tuple into a bitstring,
        replacing I -> 0 and Z -> 1.
        """
        bitstring = "".join([gate.as_bitstring for gate in self.gates])
        return int(bitstring, base=2)

    @property
    def N(self) -> int:
        """The dimension of the operator.

        This is determined by the number of gates of the operator and therefore
        equal to the number of qubits the operator acts on.
        """
        return len(self.gates)

    def sign(self, bitstring: str) -> int:
        """Sign of the bitstring to calculate the expectation value.

        The sign is dependent on the operator: It gets a factor -1 for each
        1 in the bistring if the corresponding gate is a Z gate.
        """
        gate_string = "".join([gate.as_bitstring for gate in self.gates])
        one_and_z = bin(int(bitstring, base=2) & int(gate_string, base=2))
        return (-1) ** one_and_z.count("1")


def lexicographic_ordered_operators(N: int) -> OrderedOperators:
    """All combinations of N dimensional operators, in lexicographic order."""
    # create all combinations of OrderedGates in lexicographic order
    # note that the qubit is only set temporarily, we will adjust it later
    prototypes: List[Tuple[OrderedGate, ...]]
    prototypes = list(itertools.product([IGate(-1), ZGate(-2)], repeat=N))

    operators: List[OrderedOperator] = []
    for prototype in prototypes:
        # qubit enumeration is conventionally from right to left
        qubits = range(N)[::-1]
        gates: List[OrderedGate] = []
        # adjust the qubit of each gate to the correct value
        # all gates are references to the same instance, so re-instantiate them
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

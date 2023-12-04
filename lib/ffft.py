"""Docstring."""

from typing import Optional
import warnings

import numpy as np
from sympy.ntheory import factorint

from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate, CircuitInstruction
from qiskit.circuit.library import (
    BlueprintCircuit,
    CXGate,
    CHGate,
    CZGate,
    SwapGate,
    PermutationGate,
)


class _F2Gate(Gate):
    def __init__(self, label: Optional[str] = None):
        super().__init__("F2", 2, [], label=label)

    def _define(self):
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (CXGate(), [q[0], q[1]], []),
            (CHGate(), [q[1], q[0]], []),
            (CXGate(), [q[0], q[1]], []),
            (CZGate(), [q[0], q[1]], []),
        ]

        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc


class FFFT(BlueprintCircuit):
    """Docstring"""

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        do_swaps: bool = True,
        inverse: bool = False,
        insert_barriers: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Docstring"""
        if name is None:
            name = "IFFFT" if inverse else "FFFT"

        super().__init__(name=name)
        self._do_swaps = do_swaps
        self._insert_barriers = insert_barriers
        self._inverse = inverse
        self.num_qubits = num_qubits

    @property
    def num_qubits(self) -> int:
        """Docstring"""
        # Comment
        return super().num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Docstring"""
        if num_qubits != self.num_qubits:
            self._invalidate()

            self.qregs = []
            if num_qubits is not None and num_qubits > 0:
                self.qregs = [QuantumRegister(num_qubits, name="q")]

    @property
    def insert_barriers(self) -> bool:
        """Docstring"""
        return self._insert_barriers

    @insert_barriers.setter
    def insert_barriers(self, insert_barriers: bool) -> None:
        """Docstring"""
        if insert_barriers != self._insert_barriers:
            self._invalidate()
            self._insert_barriers = insert_barriers

    @property
    def do_swaps(self) -> bool:
        """Docstring"""
        return self._do_swaps

    @do_swaps.setter
    def do_swaps(self, do_swaps: bool) -> None:
        """Docstring"""
        if do_swaps != self._do_swaps:
            self._invalidate()
            self._do_swaps = do_swaps

    def is_inverse(self) -> bool:
        """Docstring"""
        return self._inverse

    def inverse(self) -> "FFFT":
        """Docstring"""

        if self.name in ("FFFT", "IFFFT"):
            name = "FFFT" if self._inverse else "IFFFT"
        else:
            name = self.name + "_dg"

        inverted = self.copy(name=name)

        # data consists of the QFT gate only
        iffft = self.data[0].operation.inverse()
        iffft.name = name

        inverted.data.clear()
        inverted._append(CircuitInstruction(iffft, inverted.qubits, []))

        inverted._inverse = not self._inverse
        return inverted

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Docstring"""
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of qubits has not been set.")
        return valid

    def _build(self) -> None:
        """Docstring"""
        if self._is_built:
            return

        super()._build()

        num_qubits = self.num_qubits

        circuit = QuantumCircuit(*self.qregs, name=self.name)

        if num_qubits == 1:
            return

        if num_qubits == 2:
            circuit.append(_F2Gate(), qargs=self.qubits)
        else:

            circuit.i(self.qubits)

            circuit.compose(
                FFFT(num_qubits // 2), qubits=self.qubits[: num_qubits // 2], inplace=True
            )
            circuit.compose(
                FFFT(num_qubits // 2), qubits=self.qubits[num_qubits // 2 :], inplace=True
            )

            for i in range(num_qubits // 2):
                circuit.p(2 * np.pi * i / num_qubits, self.qubits[num_qubits // 2 + i])
                circuit.append(_F2Gate(), qargs=[i, num_qubits // 2 + i])
            # for i in reversed(range(num_qubits // 2)):
            #    circuit.append(_F2Gate(), qargs=[i, num_qubits // 2 + i])

        if self.do_swaps:
            pass

        wrapped = (
            # circuit.to_instruction() if self.insert_barriers else circuit.to_gate()
            circuit.to_instruction()
        )
        self.compose(wrapped, qubits=self.qubits, inplace=True)

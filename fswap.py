from typing import Optional

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister

class FSwapGate(Gate):
    """The FSWAP gate."""

    def __init__(self, label: Optional[str] = None):
        """Create new FSWAP gate."""
        super().__init__("fswap", 2, [], label=label)

    def _define(self):
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from qiskit.circuit.library import SwapGate, CZGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (SwapGate(), [q[0], q[1]], []),
            (CZGate(),   [q[0], q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(self, num_ctrl_qubits: int = 1, label: Optional[str] = None, ctrl_state=None):
        """Return controlled version of gate."""
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        """Return inverse FSWAP gate (itself)."""
        return FSwapGate()  # self-inverse

from typing import Optional

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
# from qiskit.circuit.parameterexpression import ParameterValueType

class FFFTGate(Gate):
    def __init__(self, k, label: Optional[str] = None):
        super().__init__("ffft", 2, [k], label=label)

    def _define(self):
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from qiskit.circuit.library import PhaseGate, CXGate, CHGate, CZGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (PhaseGate(2 * np.pi * self.params[0] / 8), [q[0]], []),
            (CXGate(), [q[0], q[1]], []),
            (CHGate(), [q[1], q[0]], []),
            (CXGate(), [q[0], q[1]], []),
            (CZGate(), [q[0], q[1]], []),
        ]

        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc
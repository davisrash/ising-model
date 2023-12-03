"""Bogoliubov transformation."""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

qc = QuantumCircuit(2)

angle = Parameter("θ")

def bogo(qc: QuantumCircuit, qubits: list[Qubit], theta: float) -> None:
    """Apply the Bogoliubov gate to a quantum circuit.

    Args:
        qc (QuantumCircuit): Qiskit Quantum Circuit. This circuit will be modified in place.
        qubits (list[Qubit]): List of Qubits to apply the circuits to. Often obtained with circuit.qubits
        theta (float): Angle to rotate in the X-axis by the controlled RX gate
    """

    qc.x(qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.crx(theta, qubits[0], qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.x(qubits[1])
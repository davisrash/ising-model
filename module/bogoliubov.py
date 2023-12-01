"""Bogoliubov transformation."""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

qc = QuantumCircuit(2)

angle = Parameter("θ")
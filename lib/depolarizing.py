# Code started from https://matteoacrossi.github.io/oqs-jupyterbook/project_1-solution.html
from qiskit import QuantumRegister, QuantumCircuit
import numpy as np

def depolarizing_channel( qc: QuantumCircuit, p: float, system: QuantumRegister, ancillae: list) -> QuantumCircuit:
    """Returns a copy of the quantum circuit passed in with the depolarizing channel at the end.
    Makes a copy of the circuit passed in and returns a new circuit with the oscillating gates. Input circuit is not changed.
    
    Args:
        qc (QuantumCircuit): The quantum circuit to make a copy of and put the depolarizing channel onto.
        p (float): the probability for the channel between 0 and 1.
        system (QuantumRegister): system qubit
        ancillae (list): list of QuantumRegisters for the ancillary qubits. Ancillary qubits are used for the control qubits in the order x, y, then z.

    Returns:
        A QuantumCircuit object
    """
    # Make a copy of the circuit passed in. This circuit will have the depolarizing gates added to the end and then be returned.
    dc = qc.copy()

    # Calculate the theta value which will be used to initialize the ancillary bits to the desired state.
    theta = 1/2 * np.arccos(1-2*p)

    # Initialize ancillary bits
    dc.ry(theta, ancillae[0])
    dc.ry(theta, ancillae[1])
    dc.ry(theta, ancillae[2])

    # Apply depolarizing gates
    dc.cx(ancillae[0], system)
    dc.cy(ancillae[1], system)
    dc.cz(ancillae[2], system)

    return dc

def calc_error_prob(gamma: float, t: float) -> float:
    """Calculate the value of p based on the decay rate (gamma) and time.
    Eq. p = 1 - e^( - gamma * t )

    Args:
        gamma (float): decay rate
        t (float): time

    Returns: 
        float: Probability of error
    """
    return 1 - np.exp(-1 * gamma * t)
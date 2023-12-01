"""Fourier Transform Circuit."""

from typing import Optional
import warnings
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, CircuitInstruction
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit


class Fourier(BlueprintCircuit):
    r"""Fourier Transform Circuit.

    The (fermionic?) Fourier Transform (initialism) on :math:`n` qubits is the operation

    .. math::

        |j\rangle \mapsto FFT |k\rangle

    Long explanation here.


    SEE qiskit.circuit.library.basis_change.qft.py

    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        # approximation_degree: int = 0,
        # do_swaps: bool = True,
        inverse: bool = False,
        # insert_barriers: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Construct a new (initialism) circuit.

        Args:
            num_qubits: The number of qubits on which the (initialism) acts.
            ...
            inverse: If True, the inverse Fourier transform is constructed.
            ...
            name: The name of the circuit.
        """
        if name is None:
            name = "IFFT" if inverse else "FFT"

        super().__init__(name=name)
        # self._approximation_degree = approximation_degree
        # self._do_swaps = do_swaps
        # self._insert_barriers = insert_barriers
        self._inverse = inverse
        self.num_qubits = num_qubits

    @property
    def num_qubits(self) -> int:
        """The number of qubits in the (initialism) circuit.

        Returns:
            The number of qubits in the circuit.
        """
        # This method needs to be overwritten to allow adding the setter for num_qubits while still complying to pylint.
        return super().num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits.

        Note that this changes the registers of the circuit.

        Args:
            num_qubits: The new number of qubits.
        """
        if num_qubits != self.num_qubits:
            self._invalidate()

            self.qregs = []
            if num_qubits is not None and num_qubits > 0:
                self.qregs = [QuantumRegister(num_qubits, "q")]

    # ... other properties

    def is_inverse(self) -> bool:
        """Whether the inverse Fourier transform is implemented.

        Returns:
            True, if the inverse Fourier transform is implemented, False otherwise.
        """
        return self._inverse

    def inverse(self) -> "Fourier":
        """Invert this circuit.

        Returns:
            The inverted circuit.
        """

        if self.name in ("FFT", "IFFT"):
            name = "FFT" if self._inverse else "IFFT"
        else:
            name = self.name + "_dg"

        inverted = self.copy(name=name)

        # data consists of (initialism) gate only
        ifft = self.data[0].operation.inverse()
        ifft.name = name

        inverted.data.clear()
        inverted._append(CircuitInstruction(ifft, inverted.qubits, []))

        inverted._inverse = not self._inverse
        return inverted

    # _warn_if_precision_loss

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("Number of qubits has not been set.")
        # self._warn_if_precision_loss()
        return valid

    def _build(self) -> None:
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        num_qubits = self.num_qubits

        if num_qubits == 0:
            return

        circuit = QuantumCircuit(*self.qregs, name=self.name)
        for j in reversed(range(num_qubits)):
            circuit.h(j)
            for k in reversed(range(j)):
                circuit.cp(np.pi / 2 ** (j - k), k, j)

            circuit.barrier()

        if self._inverse:
            circuit = circuit.inverse()

        wrapped = circuit.to_instruction()
        circuit.to_gate()
        self.compose(wrapped, qubits=self.qubits, inplace=True)
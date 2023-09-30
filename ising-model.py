import math

# import numpy as np

print("Hello World!")
###########################################################################################################################
#Jack's Bog transform (untested)
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *

# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options

# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(channel="ibm_quantum")

# Invoke a primitive. For more details see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials.html
# result = Sampler("ibmq_qasm_simulator").run(circuits).result()
from qiskit import QuantumCircuit, execute, Aer, IBMQ
x = QuantumRegister(2)
# Create a circuit with a register of three qubits
circ = QuantumCircuit(2)
theta = 0
# H gate on qubit 0, putting this qubit in a superposition of |0> + |1>.
circ.x(1)
# A CX (CNOT) gate on control qubit 0 and target qubit 1 generating a Bell state.
circ.cx(1,0)
#controlled rx gate
circ.crx(theta, 0, 1)
circ.cx(1,0)
circ.x(1)
circ.measure_all()
# Draw the circuit

simulator = Aer.get_backend('qasm_simulator')

# Execute the circuit on the qasm
# simulator
job = execute(circ, simulator, shots=1000)

# Grab results from the job
result = job.result()
counts = result.get_counts(circ)
print("\nTotal count for the circuit is: ",counts)
circ.draw('mpl')

################################################################################################################
#Github gates and n=4 setup
%reset -f
#%matplotlib inline
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.compiler import transpile, assemble
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.quantum_info import *
import numpy as np
provider = IBMQ.load_account()


# Loading your IBM Q account(s)
#provider = IBMQ.load_account()

#constants
n = 4
lambd = 1.2

def thetak(k,lamb):
    num = lamb - np.cos(2*np.pi*k/n)
    denom = np.sqrt( (lamb-np.cos(2*np.pi*k/n))**2 + np.sin(2*np.pi*k/n)**2)
    theta = np.arccos(num/denom)
    return theta

#Create functions based on the decomposition included in appendix of Ising paper
def bog(qcirc, q1, q2, theta):
    qcirc.x(q2)
    qcirc.cx(q2, q1)
    #Controlled RX gate qcirc.crx(theta, q1, q2) based on Jack
    qcirc.rz(np.pi/2, q2)
    qcirc.ry(theta/2, q2)
    qcirc.cx(q1, q2)
    qcirc.ry(-theta/2, q2)
    qcirc.cx(q1, q2) #changed from qc to qcirc here - Bruna
    qcirc.rz(-np.pi/2, q2)
    #####################
    qcirc.cx(q2, q1)
    qcirc.x(q2)
    qcirc.barrier()
    return qcirc

def fourier(qcirc, q1, q2, phase):
    qcirc.rz(phase, q1)
    qcirc.cx(q1, q2)
    #Controlled Hadamard
    qcirc.sdg(q1)
    qcirc.h(q1)
    qcirc.tdg(q1)
    qcirc.cx(q2, q1)
    qcirc.t(q1)
    qcirc.h(q1)
    qcirc.s(q1)
    ####################
    qcirc.cx(q1, q2)
    qcirc.cz(q1, q2)
    qcirc.barrier()
    return qcirc

def digit_sum(n):
    num_str = str(n)
    sum = 0
    for i in range(0, len(num_str)):
        sum += int(num_str[i])
    return sum

def ground_state(lamb, backend_name): # backend is now an imput, so we can plot
                                      # different ones easily - Bruna
    qc = QuantumCircuit(4, 4)
    #Set correct ground state if lambda < 1
    if lamb < 1:
        qc.x(3)
        qc.barrier()
    #magnetization
    mag = []

    #Apply disentangling gates
    qc = bog(qc, 0, 1, thetak(1.,lamb))
    qc = fourier(qc, 0, 1, 2*np.pi/n)
    qc = fourier(qc, 2, 3, 0.)
    qc = fourier(qc, 0, 1, 0.)
    qc = fourier(qc, 2, 3, 0.)
    #Set measurement step
    for i in range(0,4):
        qc.measure(i,i)

    backend = Aer.get_backend(backend_name)
    shots = 1024
    max_credits = 10 #Max number of credits to spend on execution
    job = execute(qc, backend=backend, shots=shots, max_credits=max_credits)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts(qc)
    ##while not '0000' in counts:
    #    job = execute(qc, backend=backend, shots=shots, max_credits=max_credits)
    #    job_monitor(job)
    #    result = job.result()
    #    counts = result.get_counts(qc)
    #Check what ground state is based on lambda

    r1=list(counts.keys())
    r2=list(counts.values())
    M=0
    for j in range(0,len(r1)):
        M=M+(4-2*digit_sum(r1[j]))*r2[j]/shots
    #print("$\lambda$: ",lam,", $<\sigma_{z}>$: ",M/4)
    mag.append(M/4)
    return mag

   # if lamb < 1:
   #     return counts['0001']
   # return counts['0000']/shots # it does not always works, sometimes it returns keyword error
                                # maybe we can add another else for the possibility of other states, but
                                # do not use it for plotting - Bruna


print(ground_state(lambd, 'qasm_simulator'))

#print(ground_state(.8,'statevector_simulator'))


##-----------------------------------------------------------------------------------------------------------------------------------##
# Tim's Stuff

import numpy as np
from qiskit import *
from qiskit.circuit.library import RXGate
from qiskit.circuit.library import QFT
from qiskit.circuit.library import HGate
from qiskit.circuit.library import PhaseGate

# Functions for gates

# Bogo
def bogo(circ, qubits, theta):
    circ.x(qubits[1])
    circ.cx(qubits[1], qubits[0])
    myRx = RXGate(theta=theta)
    controlledRx = myRx.control( num_ctrl_qubits=1)
    circ.append(controlledRx, qubits)
    circ.cx(qubits[1], qubits[0])
    circ.x(qubits[1])

def bogoDag(circ, qubits, theta):
    circ.x(qubits[1])
    circ.cx(qubits[1], qubits[0])
    myRx = RXGate(theta=theta)
    controlledRx = myRx.inverse().control( num_ctrl_qubits=1)
    circ.append(controlledRx, qubits)
    circ.cx(qubits[1], qubits[0])
    circ.x(qubits[1])

def qft(circuit, qubits):
    myQft = QFT(num_qubits = len(qubits))
    circuit.append(myQft, qubits)

def qftDag(circuit, qubits):
    myQft = QFT(num_qubits = len(qubits))
    myQft.inverse()
    circuit.append(myQft, qubits)


def fermionicSwap(circuit, qubits):
    circuit.swap(qubits[0], qubits[1])
    circuit.cz(qubits[0], qubits[1])

def paperFourier(circuit, qubits, n, k):
    circuit.p(2 * np.pi * k / n, qubits[0])
    circuit.cx(qubits[0], qubits[1])
    circuit.ch(qubits[1], qubits[0])
    circuit.cx(qubits[0], qubits[1])
    circtui.cz(qubits[0], qubits[1])

def paperFourierDag(circuit, qubits, n, k):
    inversePhaseGate = PhaseGate(2 * np.pi * k / n).inverse()
    circuit.append(inversePhaseGate, qubits[0:1])
    circuit.cx(qubits[0], qubits[1])
    circuit.append(HGate().inverse().control(), reversed(qubits))
    circuit.cx(qubits[0], qubits[1])
    circuit.cz(qubits[0], qubits[1])\


#Build circuit
isingCircuit = QuantumCircuit(4)
bogoDag(isingCircuit, isingCircuit.qubits[0:2], np.pi/2)
qftDag(isingCircuit, isingCircuit.qubits[0:2])
qftDag(isingCircuit, isingCircuit.qubits[2:4])
fermionicSwap(isingCircuit, isingCircuit.qubits[1:3])
qftDag(isingCircuit, isingCircuit.qubits[0:2])
qftDag(isingCircuit, isingCircuit.qubits[2:4])
#paperFourierDag(isingCircuit, isingCircuit.qubits[0:2], 2, 2)
#paperFourierDag(isingCircuit, isingCircuit.qubits[2:4], 2, 2)
fermionicSwap(isingCircuit, isingCircuit.qubits[1:3])

isingCircuit.draw('mpl')
# End of Tim's Stuff
##-----------------------------------------------------------------------------------------------------------------------------------##

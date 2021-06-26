import pennylane as qml
import numpy as numpy

dev = qml.device("default.qubit",
				 wires=["wire1", "wire2"],
				 shots=1000)

@qml.qnode(dev)
def circuit(x, y):
	qml.RZ(x, wire="wire1")
	qml.CNOT(wires=["wire1", "wire2"])
	qml.RY(y, wires="wire2")
	return qml.expval(gml.PauliZ("wire2"))

result = circuit(.543)
print(result)

dev1 = qml.device("default.qubit",
				 wires="wire3",
				 shots=1000)

dev2 = qml.device("default.qubit",
				 wires="wire4",
				 shots=1000)

dev3 = qml.device("default.qubit",
				 wires="wire5",
				 shots=1000)

@qml.qnode(dev1)
def x_rotations(params):
	qml.RX(params[0], wires=0)
	qml.RX(params[1]. wires=1)
	qml.CNOT(wires=[0, 1])
	return qml.expval(qml._PauliZ(0))

@qml.qnode(dev2)
def x_rotations(params):
	qml.RX(params[0], wires=0)
	qml.RX(params[1]. wires=1)
	qml.CNOT(wires=[0, 1])
	return qml.expval(qml.Hadamard(0))

qnodes = qml.QNodeCollection([x_rotations, y_rotations])

qnodes([0.2, 0.1])

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

dev = qml.device("default.qubit", wires=2)

theta = Parameter("▀")

qc = QuantumCircuit(2)
qc.rz(theta, [0])
qc.rx(theta, [0])
qc.cx(0, 1)

@qml.qnode(dev)
def quantum_circuit_with_loaded_subcircuit(x):
	qml.from_qiskit(qc)({theta: x})
	return qml.expval(qml.PauliZ(0))

angle = np.pi/2
result = quantum_circuit_with_loaded_subcircuit(angle)
print(result)

from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit3(phi, theta):
	qml.RX(phi[0], wires=0)
	qml.RY(phi[1], wires=1)
	qml.CNOT(wires=[0, 1])
	qml.PhaseShift(theta, wires=0)
	return qml.expval(qml.PauliZ(0))

phi = np.array([.5,.1])
theta = .2
dcircuit = qml.grad(circuit3)
dcircuit(phi, theta)
print(dir(dcircuit))


# ising-model
Transverse-field Ising model in Qiskit

## Introduction
This repository contains code for simulating a transverse-field Ising model (chain) in Qiskit. The code is primarily based on the [paper](https://doi.org/10.22331/q-2018-12-21-114) by Alba Cervera-Lierta and is written in Python 3.11.6 with Qiskit 0.45.0.

## Installation
To run the code, you need to install the packages listed in `requirements.txt`. We recommend that you create a virtual environment and use pip.

## Usage
The code runs entirely in the Jupyter notebook `ising_model.ipynb`. After installing the required packages, you may run the notebook normally. By default, the Jupyter notebook requires that you have a token from the IBM Quantum Platform saved locally, as it sends jobs through the `ibmq_qasm_simulator` available with the open plan.
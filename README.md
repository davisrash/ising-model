# ising-model
Transverse-field Ising model in Qiskit

## Introduction
This repository contains the code for the transverse-field Ising model (chain) in Qiskit. The code is based on the [paper](https://doi.org/10.22331/q-2018-12-21-114) by Alba Cervera-Lierta. The code is written in Python 3.11.6 and uses Qiskit 0.45.0.

## Installation
To run the code, you need to nstall the requirements in `requirements.txt` using pip. I recommend that you create a virtual environment.

## Usage
The code runs entirely in the Jupyter notebook `ising-model.ipynb`. After installing the required packages, you may run the notebook normally. Two things are true by default in the notebook:
1. If you have never sent a job through the IBM Quantum Experience (IBMQ) before, you will need to run the cell that saves your token. The line is commented by default.
2. The notebook will run on both a simulator (`ibmq_qasm_simulator`) and a real device (`ibm_nairobi`) by default. As of October 4, 2023, the IBMQ open (free) plan includes access to 10 min of quantum computing time per month. Comment out the lines that run on the real device if you do not want to use your time.

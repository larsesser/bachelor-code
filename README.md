# Measurement Error Mitigation Algorithm on Quantum Computers

This code is part of my Bachelor Thesis *Implementation of a Measurement Error Mitigation Algorithm on Quantum Computers*, written at the *Helmholtz-Institut für Strahlen- und Kernphysik*, University of Bonn, Germany.

The code contains an implementation of an algorithm to reduce bit-flip errors during projective measurements in the computational basis on quantum computers.
The algorithm itself was developed and published by Lena Funcke et al. in [Measurement Error Mitigation in Quantum Computers Through Classical Bit-Flip Correction](https://arxiv.org/abs/2007.03663).

The implementation is done with the open-source software development kit [Qiskit](https://github.com/Qiskit), using current version `0.34.2`.

The dependencies of the project are specified in the [Pipfile](./Pipfile).
I recommend to use the [Pipenv](https://pipenv.pypa.io/en/latest/) tool to install them.
To get a first impression of the algorithm, execute the [main.py](./main.py) file via `python3 main.py`.

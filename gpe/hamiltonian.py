from __future__ import annotations

import numpy as np

from scipy.sparse.linalg import LinearOperator

from .basis_function import WaveFunction, Density


class Hamiltonian(LinearOperator):
    N: int

    def __init__(self, N: int):
        super().__init__(dtype=np.cfloat, shape=(N, N))
        self.N = N


class Kinetic(Hamiltonian):
    pass


class Potential(Hamiltonian):
    pass


class BosonBosonCoupling(Hamiltonian):
    def __init__(self, ρ: Density):
        pass

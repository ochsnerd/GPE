from __future__ import annotations

from typing import Optional, Callable
from abc import abstractmethod, ABC

import numpy as np
import numpy.typing as npt

from scipy.sparse.linalg import LinearOperator

from .wave_function import WaveFunction, Density


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

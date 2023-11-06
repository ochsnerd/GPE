from __future__ import annotations

from typing import Optional, Callable
from abc import abstractmethod, ABC

import numpy as np
import numpy.typing as npt

from scipy.sparse.linalg import LinearOperator

from .basis_function import WaveFunction, Density


N = 5


class Hamiltonian(LinearOperator):
    N: int

    def __init__(self, N: int):
        super().__init__(dtype=np.cfloat, shape=(N, N))
        self.N = N


class Kinetic(Hamiltonian):
    pass


Δ = Kinetic(N)


class Potential(Hamiltonian):
    pass


V = Potential(N)


class BosonBosonCoupling(Hamiltonian):
    def __init__(self, ρ: Density):
        pass


BBC = BosonBosonCoupling


def solve_evp(H: Hamiltonian) -> tuple[complex, WaveFunction]:
    ...


def update(ρ_in: Density) -> tuple[complex, Density]:
    d = 0.01
    Hₙ = Δ + V + BBC(ρ_in)

    λ, ψ = solve_evp(Hₙ)

    ρ_out = ψ.compute_ρ()

    ρ_new = Density(ρ_in + d * (ρ_out - ρ_in))

    return λ, ρ_new


def iterate() -> tuple[complex, np.ndarray]:
    delta_λ = 0.0001
    delta_ρ = 0.0001

    ρ_old = Density(np.zeros(N))
    λ_old = complex(0)
    for _ in range(1000):
        λ_new, ρ_new = update(ρ_old)
        if abs(ρ_old - ρ_new) < delta_ρ or abs(λ_old - λ_new) < delta_λ:
            return solve_evp(Δ + V + BBC(ρ_new))

    raise RuntimeError("Did not converge")

from __future__ import annotations

import numpy as np

from scipy.sparse import diags, dia_array
from scipy.sparse.linalg import LinearOperator

from .wave_function import WaveFunction, Density
from .grid import Grid


class Hamiltonian(LinearOperator):
    N: int

    def __init__(self, g: Grid):
        N = g.N
        super().__init__(dtype=np.cfloat, shape=(N, N))
        self.N = N

    def expectation(self, ψ: WaveFunction) -> np.float64:
        # <ψ|ψ> = np.vdot(ψ, ψ) * grid.dx, so wrt. to np.vdot ψ isn't normalized
        return (np.vdot(ψ, self._matvec(ψ)) / np.vdot(ψ, ψ)).real


class Kinetic(Hamiltonian):
    """Implement the term -1/2 Δ using central difference.

    From the potential we know that ψ → 0 as |x| → ∞, so boundary
    conditions can be ignored as long as the domain is large enough.
    """

    K: dia_array

    def __init__(self, grid: Grid):
        super().__init__(grid)
        N = grid.N

        Δ = (
            diags(
                diagonals=[np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)],
                offsets=[-1, 0, 1],  # type: ignore
                format="dia",
                dtype=float,
            )
            / grid.dx**2
        )
        self.K = -0.5 * Δ  # type: ignore

    def _matvec(self, ψ: WaveFunction) -> WaveFunction:
        return self.K @ ψ


class Potential(Hamiltonian):
    pass


class BosonBosonCoupling(Hamiltonian):
    def __init__(self, ρ: Density):
        pass

from typing import Callable
import numpy as np
from scipy.sparse.linalg import eigsh

from .grid import Grid
from .hamiltonian import Hamiltonian, Kinetic, Potential, BosonBosonCoupling
from .wave_function import WaveFunction, Density


class Solver:
    """'General' solver, using a self-consistent field method.

    Takes as argument a function that return a Hamiltonian given a Density.

    At every step, compute the lowest eigenpair of the Hamiltonian based on
    the Density of the previous step.
    """

    d: float
    grid: Grid
    H: Callable[[Density], Hamiltonian]
    ψ_prev: WaveFunction | None

    def __init__(self, H: Callable[[Density], Hamiltonian], grid: Grid, d: float):
        self.grid = grid
        self.H = H
        self.d = d
        self.ψ_prev = None

    def solve(
        self, delta_λ: float = 0.0001, delta_ρ: float = 0.0001, max_it: int = 200
    ) -> tuple[float, WaveFunction]:

        λ_old, ρ_old = 0.0, Density(np.zeros(self.grid.N))

        for _ in range(max_it):
            λ_new, ρ_new = self._update(ρ_old)

            δρ = np.linalg.norm(ρ_old - ρ_new)
            δλ = abs(λ_old - λ_new)

            print(f"{_}:\t{λ_new=:.6f}\t({δλ=:.6f},\t{δρ=:.6f})")

            if δρ < delta_ρ and δλ < delta_λ:
                return λ_new, self.ψ_prev  # type: ignore (ψ_prev is not None)

            ρ_old, λ_old = ρ_new, λ_new

        raise RuntimeError("Did not converge")

    def _update(self, ρ_in: Density) -> tuple[float, Density]:
        λ, ψ = self._smallest_eigenpair(self.H(ρ_in))

        ρ_out = ψ.to_ρ()

        ρ_new = Density.normalize(ρ_in + self.d * (ρ_out - ρ_in), self.grid.dx)

        return λ, Density(ρ_new)

    def _smallest_eigenpair(self, H: Hamiltonian) -> tuple[float, WaveFunction]:
        """Return the smallest eigenvalue and its corresponding eigenvector."""
        # which='LM' and sigma=0 gives us the smallest eigenvalue(s) as well, and
        # is apparently "more stable". However also noticably slower
        # λs, ψs = eigsh(H, k=1, which="LM", sigma=0, v0=self.ψ_prev)
        λs, ψs = eigsh(H, k=1, which="SM", v0=self.ψ_prev)
        self.ψ_prev = WaveFunction.normalize(ψs[:, 0], self.grid.dx)
        return λs[0], self.ψ_prev

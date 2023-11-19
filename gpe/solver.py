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
    compute_H: Callable[[Density], Hamiltonian]
    H: Hamiltonian | None
    # TODO: npt.Ndarray
    ψ_prev: np.ndarray | None  # starting point for eigensolver

    def __init__(self, H: Callable[[Density], Hamiltonian], grid: Grid, d: float):
        self.grid = grid
        self.compute_H = H
        self.d = d
        self.H = None
        self.ψ_prev = None

    def solve(
        self,
        k: int = 1,
        delta_λ: float = 0.0001,
        delta_ρ: float = 0.0001,
        max_it: int = 200,
    ) -> tuple[list[float], list[WaveFunction]]:

        λ_old, ρ_old = 0.0, Density(np.zeros(self.grid.N))

        for _ in range(max_it):
            λ_new, ρ_new = self._update(ρ_old)

            δρ = np.linalg.norm(ρ_old - ρ_new)
            δλ = abs(λ_old - λ_new)

            print(f"{_}:\t{λ_new=:.6f}\t({δλ=:.6f},\t{δρ=:.6f})")

            if δρ < delta_ρ or δλ < delta_λ:
                return self._smallest_eigenpair(self.H, k=k)  # type: ignore (self.H is not None)

            ρ_old, λ_old = ρ_new, λ_new

        raise RuntimeError("Did not converge")

    def _update(self, ρ_in: Density) -> tuple[float, Density]:
        self.H = self.compute_H(ρ_in)
        λs, ψs = self._smallest_eigenpair(self.H)
        λ, ψ = λs[0], ψs[0]

        ρ_out = ψ.to_ρ()

        ρ_new = Density.normalize(ρ_in + self.d * (ρ_out - ρ_in), self.grid.dx)

        return λ, Density(ρ_new)

    def _smallest_eigenpair(
        self, H: Hamiltonian, k: int = 1
    ) -> tuple[list[float], list[WaveFunction]]:
        """Return the smallest eigenvalue and its corresponding eigenvector."""
        λs, ψs = eigsh(H, k=k, which="SA", v0=self.ψ_prev)
        self.ψ_prev = ψs[:, 0]
        return list(λs), [
            WaveFunction.normalize(ψs[:, i], self.grid.dx) for i in range(k)
        ]


class GrossPitaevskiiSolver(Solver):
    K: Kinetic
    V: Potential
    BBC: BosonBosonCoupling

    def __init__(
        self, potential: Callable[[float], float], C: float, grid: Grid, d: float = 0.01
    ):
        super().__init__(self._update_Hamiltonian, grid, d)
        self.K = Kinetic(grid)
        self.V = Potential(potential, grid)
        self.BBC = BosonBosonCoupling(Density(np.zeros(grid.N)), C, grid)

    def _update_Hamiltonian(self, ρ: Density) -> Hamiltonian:
        self.BBC = BosonBosonCoupling(ρ, self.BBC.C, self.grid)
        return self.K + self.V + self.BBC

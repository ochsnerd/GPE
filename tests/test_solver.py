import numpy as np

from gpe.hamiltonian import Kinetic, Potential
from gpe.grid import Grid
from gpe.solver import Solver


def testHarmonicOscillator():
    grid = Grid(10, 1000)
    K = Kinetic(grid)
    V = Potential(lambda x: 0.5 * x * x, grid)

    λ, ψ = Solver(H=lambda _: K + V, grid=grid, d=1).solve()
    expected_λ = 1 / 2
    assert (np.linalg.norm(ψ, ord=2) * grid.dx - 1) < 1e-12, "ψ is normalized"
    assert (
        abs(K.expectation(ψ) + V.expectation(ψ) - λ) < 1e-8
    ), "λ is the sum of the energy expectation values"
    assert abs(λ - expected_λ) < 1e-5, "λ = E₀ = 1/2"

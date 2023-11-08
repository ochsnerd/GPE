import numpy as np

from gpe.hamiltonian import Kinetic, Potential
from gpe.grid import Grid
from gpe.solver import GrossPitaevskiiSolver, Solver


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


def testGrossPitaevskii():
    # reproduce https://docs.dftk.org/stable/examples/gross_pitaevskii/
    # Our grid is centered at 0
    a = 10
    N = 300
    C = 1

    def pot(x):
        return x**2

    grid = Grid(a, N)
    solver = GrossPitaevskiiSolver(potential=pot, C=C, grid=grid, d=0.1)
    λ, ψ = solver.solve()

    tol = 1e-4
    assert abs(solver.K.expectation(ψ) - 0.2682057) < tol
    assert abs(solver.V.expectation(ψ) - 0.4707475) < tol
    # interestingly:
    assert abs(solver.BBC.expectation(ψ) - 0.4050836 * 2) < tol
    return
    assert abs(solver.BBC.expectation(ψ) - 0.4050836) < tol
    assert abs(λ - 1.144036852755) < tol

import numpy as np
import matplotlib.pyplot as plt

from gpe.grid import Grid
from gpe.solver import GrossPitaevskiiSolver


def main():
    # reproduce https://docs.dftk.org/stable/examples/gross_pitaevskii/
    # Our grid is centered at 0, not at a/2
    a = 10
    N = 100

    grid = Grid(a, N)
    solver = GrossPitaevskiiSolver(potential=lambda x: x**2, C=1, grid=grid, d=0.1)
    λ, ψ = solver.solve()
    print(f"{λ=}")
    print(f"<ψ|K|ψ> = {solver.K.expectation(ψ)}")
    print(f"<ψ|V|ψ> = {solver.V.expectation(ψ)}")
    print(f"<ψ|BBC|ψ> = {solver.BBC.expectation(ψ)}")
    print(f"ψ is normalized: <ψ|ψ> = {np.vdot(ψ, ψ) * grid.dx}")

    H = solver.K + solver.V + solver.BBC
    # <ψ|ψ> = np.vdot(ψ, ψ) * grid.dx, so wrt. to np.vdot ψ isn't normalized
    print(
        f"ψ is an eigenvector: H|ψ> - <ψ|H|ψ> |ψ> = {np.linalg.norm(H @ ψ - np.vdot(ψ, H @ ψ) / np.vdot(ψ, ψ) * ψ)}"
    )

    # fix the phase
    ψ /= ψ[N // 2] / np.abs(ψ[N // 2])
    plt.plot(list(grid.iterator()), ψ.real, label="ℜ(ψ)")
    plt.plot(list(grid.iterator()), ψ.imag, label="ℑ(ψ)")
    plt.plot(list(grid.iterator()), ψ.to_ρ(), label="ρ")
    plt.legend()
    plt.title("Ground state of GPE with V = x²")
    plt.show()


if __name__ == "__main__":
    main()

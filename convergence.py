import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from gpe.grid import Grid
from gpe.solver import GrossPitaevskiiSolver


def create_solver(N: int) -> GrossPitaevskiiSolver:
    C = 1
    a = 10

    def pot(x):
        return x**2

    return GrossPitaevskiiSolver(potential=pot, C=C, grid=Grid(a, N), d=0.1)


def main():
    base = 2
    ref_N_exp = 10
    λ_ref, _ = create_solver(base**ref_N_exp).solve()

    Ns = [2**i for i in range(4, int(ref_N_exp))]
    λs = []
    for N in Ns:
        λ, _ = create_solver(N).solve()
        λs.append(abs(λ_ref - λ))

    plt.plot(Ns, λs, label=f"|λ(N) - λ({base**ref_N_exp})|")
    plt.plot(
        Ns,
        [(N / Ns[0]) ** (-2) * λs[0] for N in Ns],
        linestyle="dashed",
        color="grey",
        label=r"$\mathcal{O}(N^{-2})$",
    )
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xlabel("N")
    plt.legend()
    plt.title("Convergence")
    plt.show()


if __name__ == "__main__":
    main()

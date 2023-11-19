from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from gpe.solver import GrossPitaevskiiSolver
from gpe.grid import Grid


@dataclass
class Potential:
    β: float
    α1: float
    α2: float
    L1: float
    L2: float
    R1: float
    R2: float

    _pref1: float = field(init=False)  # -α1 / L1 / √2 / √π
    _pref2: float = field(init=False)  # -α2 / L2 / √2 / √π

    def __post_init__(self):
        self._pref1 = self._pref(self.α1, self.L1)
        self._pref2 = self._pref(self.α2, self.L2)

    def __call__(self, x: float) -> float:
        return (
            self._G(self._pref1, self.R1, self.L1, x)
            + self._G(self._pref2, self.R2, self.L2, x)
            + self.β * x * x
        )

    def _G(self, pref: float, R: float, L: float, x: float) -> float:
        return pref * np.exp(-0.5 * ((x - R) / L) ** 2)

    def _pref(self, α: float, L: float) -> float:
        return -α / np.sqrt(2 * np.pi) / L


def main():
    V = Potential(
        β=1 / 10,
        α1=5,
        α2=5,
        L1=1,
        L2=1,
        R1=3,
        R2=-3,
    )

    grid = Grid(
        a=15,
        N=101,
    )

    solver = GrossPitaevskiiSolver(potential=V, C=1, grid=grid, d=0.05)
    λs, ψs = solver.solve(k=5, max_it=500)

    # energies
    for i, (λ, ψ) in enumerate(zip(λs, ψs)):
        print(f"λ{i+1}:")
        print(f"\tλ{i+1} = {λ}")
        print(f"\t<ψ|K|ψ> = {solver.K.expectation(ψ)}")
        print(f"\t<ψ|V|ψ> = {solver.V.expectation(ψ)}")
        print(f"\t<ψ|BBC|ψ> = {solver.BBC.expectation(ψ)}")

    xs = list(grid.iterator())

    # total potential
    Vs = [V(x) for x in xs]
    bbc = solver.BBC
    plt.plot(xs, Vs, color="grey", label="$V$")
    plt.plot(
        xs, [Vi + 2 * bbc.C * ρi for Vi, ρi in zip(Vs, bbc.ρ)], label=r"$V + 2C\rho$"
    )
    plt.legend(loc="lower right")
    plt.title(r"Total potential $V + 2C\rho$")
    plt.show()

    # densities
    for i, (λ, ψ) in enumerate(zip(λs, ψs)):
        plt.plot(xs, ψ.to_ρ(), label=f"$ρ_{{{i}}}$")

    V_max, ψ_max = max(Vs), max(max(ψ.to_ρ()) for ψ in ψs)
    plt.plot(xs, [Vi / V_max * ψ_max for Vi in Vs], color="grey", label=r"$\propto V$")
    plt.legend(loc="lower right")
    plt.title("Lowest eigenstates of GPE")
    plt.show()


if __name__ == "__main__":
    main()

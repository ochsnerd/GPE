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
        N=151,
    )

    solver = GrossPitaevskiiSolver(potential=V, C=1, grid=grid, d=0.1)
    λ, ψ = solver.solve()

    xs = list(grid.iterator())
    ψ /= ψ[grid.N // 2] / np.abs(ψ[grid.N // 2])
    plt.plot(xs, ψ.real, label="ℜ(ψ)")
    plt.plot(xs, ψ.imag, label="ℑ(ψ)")
    plt.plot(xs, ψ.to_ρ(), label="ρ")
    Vs = [V(x) for x in xs]
    V_max, ψ_max = max(Vs), max(ψ.real)
    plt.plot(xs, [Vi / V_max * ψ_max for Vi in Vs], color="grey", label="V(x)")
    plt.legend(loc="upper right")
    plt.title("Ground state of GPE")
    plt.show()


if __name__ == "__main__":
    main()

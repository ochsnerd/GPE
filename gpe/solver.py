import numpy as np
from scipy.sparse.linalg import eigsh

from .hamiltonian import Hamiltonian, Kinetic, Potential, BosonBosonCoupling
from .wave_function import WaveFunction, Density


N = 5
Δ = Kinetic(N)
V = Potential(N)
BBC = BosonBosonCoupling


def solve_evp(H: Hamiltonian) -> tuple[float, WaveFunction]:
    λ, ψ = eigsh(Hamiltonian, k=1)
    return λ, WaveFunction(ψ)


def update(ρ_in: Density) -> tuple[complex, Density]:
    d = 0.01
    Hₙ = Δ + V + BBC(ρ_in)

    λ, ψ = solve_evp(Hₙ)

    ρ_out = ψ.compute_ρ()

    ρ_new = Density(ρ_in + d * (ρ_out - ρ_in))

    return λ, ρ_new


def iterate() -> tuple[complex, WaveFunction]:
    delta_λ = 0.0001
    delta_ρ = 0.0001

    ρ_old = Density(np.zeros(N))
    λ_old = complex(0)
    for _ in range(1000):
        λ_new, ρ_new = update(ρ_old)
        if abs(ρ_old - ρ_new) < delta_ρ or abs(λ_old - λ_new) < delta_λ:
            return solve_evp(Δ + V + BBC(ρ_new))

    raise RuntimeError("Did not converge")

import numpy as np

from gpe.wave_function import Density, WaveFunction


def test_to_ρ():
    def ψ_to_ρ(ψ: list[complex], ρ: list[float]):
        assert np.allclose(
            WaveFunction(np.array(ψ, dtype=np.cfloat)).to_ρ(),
            Density(np.array(ρ, dtype=float)),
        )

    ψ_to_ρ([1, 0], [1, 0])
    ψ_to_ρ([1j, 0], [1, 0])
    ψ_to_ρ([1j, -1j], [1, 1])
    ψ_to_ρ([1 + 1j, 1 - 1j], [2, 2])


def test_Normalization():
    def ρ_norm(ρ, dx):
        return sum(ρ) * dx

    def ψ_norm(ψ, dx):
        return sum(ψ * np.conjugate(ψ)) * dx

    dx = 0.1
    vecs = [
        [1j, 2, 3, 4],
        [1 - 1j],
        np.concatenate((np.arange(1000) * 1j, np.arange(1000))),
    ]
    for vec in vecs:
        ρ = Density(np.array(vec).real)
        assert abs(ρ_norm(Density.normalize(ρ, dx), dx) - 1) < 1e-12
        ψ = WaveFunction(np.array(vec))
        assert abs(ψ_norm(WaveFunction.normalize(ψ, dx), dx) - 1) < 1e-12

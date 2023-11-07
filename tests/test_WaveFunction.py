import numpy as np

from gpe.wave_function import Density, WaveFunction


def test_ρ():
    def ψ_to_ρ(ψ: list[complex], ρ: list[float]):
        assert np.allclose(
            WaveFunction(np.array(ψ, dtype=np.cfloat)).to_ρ(),
            Density(np.array(ρ, dtype=float)),
        )

    ψ_to_ρ([1, 0], [1, 0])
    ψ_to_ρ([1j, 0], [1, 0])
    ψ_to_ρ([1j, -1j], [1, 1])
    ψ_to_ρ([1 + 1j, 1 - 1j], [2, 2])

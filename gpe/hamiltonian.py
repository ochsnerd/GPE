from __future__ import annotations

from typing import Optional, Callable
from abc import abstractmethod, ABC

import numpy as np
import numpy.typing as npt

from scipy.sparse.linalg import LinearOperator

from .wave_function import WaveFunction, Density

# (Analogously to https://docs.dftk.org/stable/guide/periodic_problems/#Periodic-operators-and-the-Bloch-transform)
# Factorizing ψₖ(x) = eⁱᵏˣuₖ(x), where uₖ(x) is a lattice-periodic function,
# we can show that solving the EVP Hψ = λψ for
# H = -½Δ + V + 2Cρ
# is equivalent to solving the EVP Hₖuₖ = λuₖ for
# Hₖ = ½(-i∇ +k)² + V + 2C e⁻ⁱᵏˣ uₖ* ∀k,
# where uₖ* is the complex conjugate of uₖ.
# In the following, we restrict the problem the a periodic box of size a.
# In this box, we choose a plane-wave basis, i.e.
# uₖ(x) = Σ c_G e_G(x)
# with
# e_G(x) = eⁱᴳˣ / √a
# a plane wave, and the summation running over
# G ∈ {G | (G + k)² ≤ 2 E_C}
# where E_C is a cutoff-energy.
# Also we set k = 2π/a.
#
# TODO:
# - I just silently dropped the n-subscript. I think it comes from the periodicy
#   and can be ignored since we're just working in a single lattice-cell?
# - I don't have a good motivation for choosing k. I think that comes from the
#   fact that I don't fully understand what's going on.
# - I'm not doing anything with Fourier-transforms. I think one application would be
#   that the momentum operator is diagonal in Momentum-space, so changing to that basis
#   with an FFT makes computing that term more efficient. However I'm don't know if the
#   FFT is used for the same reason in the DFTK.


class Hamiltonian(LinearOperator):
    N: int

    def __init__(self, N: int):
        super().__init__(dtype=np.cfloat, shape=(N, N))
        self.N = N


class Kinetic(Hamiltonian):
    """Implement the term ½(-i∇ +k)²"""

    def __init__(self, N: int, k: float):
        super().__init__(N)
        self.k = k


class Potential(Hamiltonian):
    pass


class BosonBosonCoupling(Hamiltonian):
    def __init__(self, ρ: Density):
        pass

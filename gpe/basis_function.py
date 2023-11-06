from __future__ import annotations

import numpy as np

class WaveFunction(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        # see https://numpy.org/doc/stable/user/basics.subclassing.html
        assert input_array.dtype == np.cfloat
        return np.asarray(input_array).view(cls)

    # what is the relation to the basisfunction?
    def compute_ρ(self) -> Density:
        # some kind of inner product
        ...


class Density(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        # see https://numpy.org/doc/stable/user/basics.subclassing.html
        assert input_array.dtype == float
        return np.asarray(input_array).view(cls)



# Not sure if this is needed, just use implicitly PaneWave?
# If extendible: Give Wavefunction (and Density for safety, not actually needed I think)
# a Basis, then delegate operations by Hamiltonians to the Basis
class Basis:
    # abc?
    pass


class PlaneWaveBasis(Basis):
    def __init__(self, k: float, period: float):
        pass

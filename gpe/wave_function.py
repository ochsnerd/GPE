from __future__ import annotations

import numpy as np


class WaveFunction(np.ndarray):
    """Stores the coefficients of the straightforward real-space discretization."""

    def __new__(cls, input_array: np.ndarray):
        # see https://numpy.org/doc/stable/user/basics.subclassing.html
        # We can't really use the normal python __init__, because
        # ndarrays can be constructed from other ways.
        assert len(input_array.shape) == 1
        assert input_array.dtype == np.cfloat
        return np.asarray(input_array).view(cls)

    def to_ρ(self) -> Density:
        return Density(np.multiply(self, np.conjugate(self)).real)

    # def normalize(self) -> None:
    #     how to assign to self?


class Density(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        # see https://numpy.org/doc/stable/user/basics.subclassing.html
        assert len(input_array.shape) == 1
        assert input_array.dtype == float
        return np.asarray(input_array).view(cls)

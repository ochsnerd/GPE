import numpy as np
from scipy.sparse.linalg import eigsh

from gpe.hamiltonian import Hamiltonian


class M(Hamiltonian):
    # Implement the Matrix mat

    def __init__(self, mat):
        assert len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]
        super().__init__(mat.shape[0])
        self.mat = mat
        self.ops += [lambda x: self.mat @ x]


def ev_equation_residual(A, eigenvalue, eigenvector):
    return np.linalg.norm(A @ eigenvector - eigenvalue * eigenvector)


def check_eigval_eigvec(m, tol=1e-12):
    assert ev_equation_residual(m, *eigsh(M(m), k=1)) < tol


def testEVP():
    matrices = [
        np.array(
            [[0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]]),
        np.array(
            [[1j, 0, 1j],
            [0, 1, 0],
            [-1j, 0, -1j]]),
        np.array(
            [[1.1, -2j, 3, 4j],
            [2j, 3.3, -4j, 5],
            [3, 4j, 5.5, -6j],
            [-4j, 5, 6j, 7.7]]
        )
    ]

    for m in matrices:
        check_eigval_eigvec(m)

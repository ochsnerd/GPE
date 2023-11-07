import numpy as np
from scipy.sparse.linalg import eigsh
from gpe.grid import Grid

from gpe.hamiltonian import Hamiltonian, Kinetic


class M(Hamiltonian):
    # Implement the Matrix mat

    def __init__(self, mat):
        assert len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]
        super().__init__(Grid(1, mat.shape[0]))
        self.mat = mat

    def _matvec(self, x):
        return self.mat @ x


def ev_equation_residual(A, eigenvalue, eigenvector):
    return np.linalg.norm(A @ eigenvector - eigenvalue * eigenvector)


def check_eigenvalue_equation_holds(matrix, operator, tol=1e-12):
    assert ev_equation_residual(matrix, *eigsh(operator, k=1)) < tol


def testEVP():
    matrices = [
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        np.array([[1j, 0, 1j], [0, 1, 0], [-1j, 0, -1j]]),
        np.array(
            [[1.1, -2j, 3, 4j], [2j, 3.3, -4j, 5], [3, 4j, 5.5, -6j], [-4j, 5, 6j, 7.7]]
        ),
    ]

    for m in matrices:
        check_eigenvalue_equation_holds(m, M(m))


def testHamiltonian_addition():
    m1 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    m2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    m3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    check_eigenvalue_equation_holds(m1 + m2 + m3, M(m1) + M(m2) + M(m3))


def testKinetic():
    g = Grid(3, 4)  # -> dx = 1
    # which='LM' and sigma=0 gives us the smallest eigenvalue(s)
    λ, _ = eigsh(Kinetic(g), k=1, which="LM", sigma=0)
    # λ, _ = eigsh(Kinetic(g), k=1, which="SM")
    assert np.abs(λ[0] - 0.25 * (3 - np.sqrt(5))) < 1e-12

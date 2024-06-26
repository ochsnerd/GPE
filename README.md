# Problem

The Gross-Pitaevskii equation in 1D reads

$$H\psi = \left( -\frac{1}{2} \Delta + V + 2C \lvert \psi \rvert^2 \right) \psi = \lambda \psi \quad \Vert\psi\Vert_{L2} = 1.$$

This repo attempts to solve the problems stated in https://matmat.org/jobs/problem/.

# Setup

Install packages:
``` sh
python3.12 -m pip install -r requirements.txt  # lower versions of python might work as well
```

Run tests:

``` sh
python3.12 -m pytest
```

# Implementation

The `GrossPitaevksiiSolver` in `gpe/solver.py` solves the Gross-Pitaevskii equation using the
SCF procedure from the problem description. It does so using `Hamiltonian` and `WaveFunction` objects,
which represent the problem in real space. The kinetic energy is computed with a straighforward central
difference approximation, the potential and the boson-boson coupling are element-wise multiplications with
the discretized wave function. The eigenvalue problem at each step is solved by
[`scipy.sparse.linalg.eigsh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html),
which wraps an Lanczos-type algorithm from ARPACK.

## Extensions

Currently, the Laplace operator in the kinetic energy term is implemented as a sparse matrix-vector
multiplication, an operation with time-complexity $\mathcal{O}(N)$ (citation needed), where $N$ is
the system size (and crucially the number of nonzero terms in the matrix). Also it has a truncation
error of $\mathcal{O}(\Delta x^2) = \mathcal{O}(N^{-2})$. This means the work to reduce the error scales
as $\mathcal{O}(N^3)$.
Of course this is only the first step in analyzing the algorithm, since the iterative eigensolver and the
SCF procedure are built on top of this.

In contrast, transforming the wavefunction into momentum-space via FFT costs $\mathcal{O}(\log N)$, and
allows us to compute the kinetic energy as a simple element-wise multiplication. Since
the boson-boson interaction still has to be applied in real space, the wavefunction has
to be transformed back.

I don't know:
- Does using the FFT as described allow us to diagonalize $H$ in a more efficient way than the iterative eigensolver used currently?
- What is the discretization error associated with applying the kinetic operator in Fourier space?
- What are the effects of implementing this strategy in code,
  e.g. how explicitly the plane waves are present in the implementation (are they just implicitly multplied
  with the coefficients in the Fourier basis, or do they show up explicitly somewhere), 
  and how is this related to the Hamiltonian blocks $H_k$ described in the
  [DFTK docs](https://docs.dftk.org/stable/guide/periodic_problems/#Discretization-and-plane-wave-basis-sets).

# Reproduce DFTK GPE Example

`python3.12 reproduce_DFTK_GPE.py` computes the solution to the GPE with $V(x) = x^2$.

Visually, the resulting $\psi$ and $\rho$ match the expected solution from the [DFTK GPE example](https://docs.dftk.org/stable/examples/gross_pitaevskii/):

![rho psi](figs/GPE_parabola.png "rho psi")

Similarly, the energy contribution of the kinetic and potential term match (up to expected
discretization errors). However, the energy of the boson-boson interaction is off by a factor of 2.
This is independent of the grid spacing and the cutoff-distance (provided it is large enough to ignore
the boundary conditions).

| $[Ha]$ | My Solution | DFTK |
|--|-------------|------|
| $\bra{\psi}H\ket{\psi}$ | 1.5490696  |  1.1440368  |
| $\bra{\psi}K\ket{\psi}$ | 0.2680410  |  0.2682057  |
| $\bra{\psi}V\ket{\psi}$ | 0.4704457  |  0.4707475  |
| $\bra{\psi}(2C\rho)\ket{\psi}$ | 0.8105828  | 0.4050836  |

# Convergence

See below the error of the resulting ground state energy of the GPE with parabolic potential (same as above), i.e. $|\lambda_{[N]} - \lambda_{ref}|$,
where $\lambda_{[N]}$ is computed on a grid with $N$ points and $\lambda_{ref}$
is the reference solution computed on grid with 1024 gridpoints.

![convergence plot](figs/convergence.png "convergence plot")

From the discretization error of the kinetic error we expect a convergence rate of $\mathcal{O}(N^{-2})$. The deviations of the observed error from this scaling can be attributed to the
iterative eigensolver and the SCF procedure.

# Task 2: Two-Gaussian Potential

`python3.12 gaussian_pot.py` computes the solution to the GPE with the two-gaussian potential given in the problem description. The following discussion assumes the setting given in the problem statement
($C=1, \beta=1/10, L=L^\prime=1, \alpha=\alpha^\prime=5, R=R^\prime=3)$.

The total potential looks like this:

![total potential plot](figs/total_potential.png "total potential plot")

The lowest energies are as follows:

```
λ1:
	λ1 = -0.2624610991767509
	<ψ|K|ψ> = 0.1459762191927695
	<ψ|V|ψ> = -0.7284081213446412
	<ψ|BBC|ψ> = 0.31997080297512076
λ2:
	λ2 = -0.22460667828721137
	<ψ|K|ψ> = 0.23785684748793545
	<ψ|V|ψ> = -0.8068677364564136
	<ψ|BBC|ψ> = 0.34440421068126753
λ3:
	λ3 = 0.28564534384344503
	<ψ|K|ψ> = 0.40131566036432037
	<ψ|V|ψ> = -0.28315851422732813
	<ψ|BBC|ψ> = 0.16748819770645612
λ4:
	λ4 = 0.6724420668953223
	<ψ|K|ψ> = 0.7980588638020367
	<ψ|V|ψ> = -0.3272687209904828
	<ψ|BBC|ψ> = 0.2016519240837733
λ5:
	λ5 = 1.1496357542577598
	<ψ|K|ψ> = 1.1519325318907396
	<ψ|V|ψ> = -0.1907683744487604
	<ψ|BBC|ψ> = 0.18847159681578338
```

(note that these overstate the boson-boson couling by a factor of two)

The ground states are:

![ground states](figs/eigenstates.png "ground states")


# Task 3:

Solve the optimization problem $$\min_{L, L^\prime, \alpha, \alpha^\prime, R, R^\prime \in \mathbb{R}} \lvert 10\lvert\lambda_2 - \lambda_1\rvert - \lvert\lambda_3 - \lambda_2\rvert \rvert,$$
where we obtain $\lambda_1, \lambda_2, \lambda_3$ by solving the GPE with Hamiltonian parameters $L, L^\prime, \alpha, \alpha^\prime, R$ and $R^\prime$.
Taking the hint into account reduces the dimensionality of the problem from 6 to 1.
Probably the search-space can be restricted somewhat due to the parabolic potential term invalidating Gaussians with large $R$ (as long as $\alpha$ is not too big).

- Nelder-Mead (derivative-free general purpose optimization)
- Simulated annealing (derivative-free general purpose optimization)
  TODO: explain more
- Stochastic gradient descent
  - stochastic because the problem will not have a unique minimum
  - we don't have the derivative readily available, and computing it via discretization is very costly
- TODO: can this be formulated as a Bayesian inverse problem? Would this differ on an implementation level from simulated annealing

``` python
from gaussians_pot import Potential

def f(L1: float, L2: float, α1: float, α2: float, R1: float, R2: float) -> float:
    V = Potential(
        β=1 / 10,
        α1=α1,
        α2=α2,
        L1=L1,
        L2=L2,
        R1=R1,
        R2=R2,
    )

    grid = Grid(
        a=15,  # make sure this is large enough to prevent boundary effects
        N=101,
    )
    
    (λ1, λ2, λ3), _ = GrossPitaevksiiSolver(potential=V, C=1, grid=grid).solve(k=3)
    
    return abs(10 * abs(λ2 - λ1) - abs(λ3 - λ2))


some_solver.minimize(f)
```

I expect the code as it is (in python) to be too slow for such an approach.

(Funninly enough, because of my error where the boson-boson coupling contribution to the energy is too large, the solution with the standard parameters is already really close to the 1:10 ratio.)

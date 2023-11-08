class Grid:
    """Real space grid.

    Centered at 0, length a, N gridpoints.
    Grid[i] = xᵢ
    Grid[0] = -a/2
    rid[N-1] = a/2
    """

    def __init__(self, a: float, N: int):
        self.a = a
        self.N = N
        self.dx = a / (N - 1)

    def __getitem__(self, i: int) -> float:
        """Get the value of the i-th gridpoint."""
        if not isinstance(i, int):
            raise TypeError
        assert 0 <= i < self.N
        return i * self.dx - self.a / 2

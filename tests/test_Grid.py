from gpe.grid import Grid


def testGrid():
    x = Grid(2, 101)
    assert x[0] == -1
    assert x[100] == 1
    assert x[50] == 0

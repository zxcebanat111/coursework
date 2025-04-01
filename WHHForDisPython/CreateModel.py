from math import sqrt


def CreateModel(x, y):
    n = len(x)
    d = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            d[i][j] = sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            d[j][i] = d[i][j]
    model = {
             "n" : n,
             "x" : x,
             "y" : y,
             "d" : d,
             }
    return model

import numpy as np


def TourLength(position,
               distances):
    tour = position.copy()
    n: int = len(tour)
    tour = np.append(tour, tour[0])
    L: float = 0.0
    for k in range(n):
        i = tour[k]
        j = tour[k + 1]
        L += distances[i][j]
    return L

from typing import List
import numpy as np
import numpy.typing as npt


def RouletteWheelSelection(P: List[float] | npt.NDArray[np.float64]):
    r: float = np.random.rand()
    c: npt.NDArray[np.float64] = np.cumsum(P)
    j: int = int(np.searchsorted(c, r, 'left'))
    return j

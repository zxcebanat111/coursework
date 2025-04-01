from typing import Self
import numpy as np
import numpy.typing as npt


class Solution:

    def __init__(self,
                 Position: npt.NDArray[np.int64] = np.array([]),
                 Cost: float = np.inf):

        self.Position = Position
        self.Cost = Cost

    def __lt__(self, other):
        return self.Cost < other.Cost

    def copy(self) -> Self:
        return self.__class__(self.Position, self.Cost)

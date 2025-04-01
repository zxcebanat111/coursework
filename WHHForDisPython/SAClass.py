from typing import Any, Callable
import numpy as np
import numpy.typing as npt

from RouletteWheelSelection import RouletteWheelSelection
from SolutionClass import Solution


class SA:

    def __init__(self,
                 maxSubIt: int  = 10,
                 T0: float      = 0.025,
                 alpha: float   = 0.99,
                 nNeigbors: int = 5):

        self.maxSubIt: int  = maxSubIt
        self.T: float      = T0
        self.alpha: float   = alpha
        self.nNeigbors: int = nNeigbors


    def initialize(self,
                   population: npt.NDArray[Any],
                   costFunction: Callable[[npt.NDArray[Any]], float]) \
    -> Solution:

        
        nPopulations: int = population.size
        bestSol: Solution = Solution()
        for _ in range(self.maxSubIt):
            newPopulations: npt.NDArray[Any]  = np.array([[Solution()
                                                                for _ in range(self.nNeigbors)]
                                                               for _ in range(nPopulations)])
            for i in range(nPopulations):
                for j in range(self.nNeigbors):
                    # for sol in population:
                    #     print("n:", sol.Position.size)
                    #     if sol.Position.size != 20:
                    #         exit(0)
                    newPopulations[i][j].Position = self.CreateNeighbor(population[i].Position)

                    newPopulations[i][j].Cost = costFunction(newPopulations[i][j].Position)

            newPopulation: npt.NDArray[Any] = np.sort(newPopulations.flatten())

            for i in range(nPopulations):
                if newPopulation[i].Cost <= population[i].Cost:
                    population[i] = newPopulation[i].copy()
                else:
                    delta: float = (newPopulation[i].Cost - population[i].Cost) / population[i].Cost
                    P: float = np.exp(-delta / self.T)
                    if np.random.rand() <= P:
                        population[i] = newPopulation[i].copy()

                if population[i].Cost <= bestSol.Cost:
                    bestSol = population[i].copy()
        
        return bestSol


    def CreateNeighbor(self,
                       position: npt.NDArray[np.int64],
                       pSwap: float = 0.2,
                       pReversion: float = 0.5,
                       pInsertion: float = 0.3):

        def ApplySwap(pos: npt.NDArray[np.int64],):
            n: int = pos.size
            i, j = np.random.choice(n, 2, replace=False)
            newPos: npt.NDArray[np.int64] = pos.copy()
            newPos[i], newPos[j] = newPos[j], newPos[i]
            return newPos

        def ApplyReversion(pos: npt.NDArray[np.int64],):
            n: int = len(pos)
            i, j = np.random.choice(n, 2, replace=False)
            if j < i:
                i, j = j, i
            newPos = np.concatenate([pos[:i], pos[i:j][::-1], pos[j:]], axis=0)
            return newPos


        def ApplyInsertion(pos: npt.NDArray[np.int64],):
            n: int = len(pos)
            i, j = np.random.choice(n, 2, replace=False)
            if i < j:
                newPos = np.concatenate([pos[:i], pos[i + 1:j + 1], [pos[i]], pos[j+1:]])
            else:
                newPos = np.concatenate([pos[:j+1], [pos[i]], pos[j+1:i], pos[i+1:]])
            return newPos 


        pSwap = 1 - pInsertion - pReversion
        pReversion = 1 - pSwap - pInsertion
        pInsertion = 1 - pSwap - pReversion

        p = [pSwap, pReversion, pInsertion]

        Method: int = RouletteWheelSelection(p)

        # Swap
        if Method == 0:
            newPosition = ApplySwap(position)
        # Reversion
        elif Method == 1:
            newPosition = ApplyReversion(position)
        # Inseriton
        else:
            newPosition = ApplyInsertion(position)
        return newPosition

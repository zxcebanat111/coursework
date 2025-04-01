from typing import Any, Callable, Tuple
import numpy as np
import numpy.typing as npt

from RouletteWheelSelection import RouletteWheelSelection
from SolutionClass import Solution


class ACO:
    def __init__(self,
                 nAnts: int     = 50,
                 alpha: float   = 1.0,
                 beta: float    = 1.0,
                 rho: float     = 0.05):
        
        self.nAnts: int    = nAnts
        self.alpha: float  = alpha
        self.beta: float   = beta
        self.rho: float    = rho

    def initialize(self,
                   population: npt.NDArray[Any],
                   costFunction: Callable[[npt.NDArray[Any]], float]) \
    -> Solution:
        
        nPopulations: int = population.size
        nVar: int = population[0].Position.size
        bestSol: Solution = Solution()
        
        tau: npt.NDArray[np.float64] = np.ones((nVar, nVar)) * 0.1
        
        antSolutions: npt.NDArray[Any] = np.array([Solution() for _ in range(self.nAnts)])
        
        for ant in range(self.nAnts):
            currentPos: npt.NDArray[np.int64] = population[np.random.randint(nPopulations)].Position.copy()
            antSolutions[ant].Position = self.constructSolution(currentPos, tau)
            antSolutions[ant].Cost = costFunction(antSolutions[ant].Position)
        
        for antSol in antSolutions:
            if antSol.Cost <= bestSol.Cost:
                bestSol = antSol.copy()
        
        tau = (1 - self.rho) * tau
        
        for antSol in antSolutions:
            deltaTau: float = 1.0 / (antSol.Cost + 1e-10)
            pos: npt.NDArray[np.int64] = antSol.Position
            for i in range(nVar-1):
                tau[pos[i], pos[i+1]] += deltaTau
            tau[pos[-1], pos[0]] += deltaTau
        
        sortedAnts: npt.NDArray[Any] = np.sort(antSolutions)
        for i in range(min(nPopulations, self.nAnts)):
            population[i] = sortedAnts[i].copy()
        
        return bestSol

    def constructSolution(self,
                         position: npt.NDArray[np.int64],
                         tau: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        
        nVar: int = position.size
        newPosition: npt.NDArray[np.int64] = position.copy()
        visited: set = set()
        
        current: int = np.random.randint(nVar)
        newPosition[0] = current
        visited.add(current)
        
        for i in range(1, nVar):
            probs: npt.NDArray[np.float64] = np.zeros(nVar)
            for j in range(nVar):
                if j not in visited:
                    heuristic: float = 1.0 / (abs(newPosition[i-1] - j) + 1e-10)
                    probs[j] = (tau[newPosition[i-1], j] ** self.alpha) * (heuristic ** self.beta)
            
            probs_sum: float = probs.sum()
            if probs_sum > 0.0:
                probs /= probs_sum
            
            next_pos: int = RouletteWheelSelection(probs.tolist())
            newPosition[i] = next_pos
            visited.add(next_pos)
        
        return newPosition

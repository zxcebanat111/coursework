from typing import Any, Callable
import numpy as np
import numpy.typing as npt

from SolutionClass import Solution


class GWO:
    def __init__(self,
                 nWolves: int   = 50,
                 maxIt: int     = 100,
                 aMax: float    = 2.0,
                 aMin: float    = 0.0):
        
        self.nWolves: int = nWolves
        self.maxIt: int   = maxIt
        self.aMax: float  = aMax
        self.aMin: float  = aMin

    def initialize(self,
                   population: npt.NDArray[Any],
                   costFunction: Callable[[npt.NDArray[Any]], float]) \
    -> Solution:
        
        nPopulations: int = population.size
        nVar: int = population[0].Position.size
        bestSol: Solution = Solution()
        
        wolves: npt.NDArray[Any] = population.copy()
        for i in range(nPopulations):
            wolves[i].Cost = costFunction(wolves[i].Position)
        
        for i in range(nPopulations):
            if not self.isValidPermutation(wolves[i].Position):
                wolves[i].Position = np.random.permutation(nVar)
                wolves[i].Cost = costFunction(wolves[i].Position)
        
        wolves = np.sort(wolves)
        
        for it in range(self.maxIt):
            a: float = self.aMax - (self.aMax - self.aMin) * (it / self.maxIt)
            
            alpha: Solution = wolves[0].copy()
            beta: Solution = wolves[1].copy()
            delta: Solution = wolves[2].copy() if self.nWolves >= 3 else wolves[1].copy()
            
            for i in range(self.nWolves):
                wolf: Solution = wolves[i]
                
                newPos: npt.NDArray[np.int64] = self.updatePosition(wolf.Position, alpha.Position, beta.Position, delta.Position, a)
                
                if not self.isValidPermutation(newPos):
                    newPos = self.correctPermutation(newPos, nVar)
                
                wolf.Position = newPos
                wolf.Cost = costFunction(wolf.Position)
            
            wolves = np.sort(wolves)
            
            if wolves[0].Cost < bestSol.Cost:
                bestSol = wolves[0].copy()
        
        for i in range(min(nPopulations, self.nWolves)):
            population[i] = wolves[i].copy()
        
        return bestSol

    def updatePosition(self,
                      currentPos: npt.NDArray[np.int64],
                      alphaPos: npt.NDArray[np.int64],
                      betaPos: npt.NDArray[np.int64],
                      deltaPos: npt.NDArray[np.int64],
                      a: float) -> npt.NDArray[np.int64]:
        
        nVar: int = currentPos.size
        
        currentOrder: npt.NDArray[np.float64] = np.argsort(currentPos).astype(np.float64)
        alphaOrder: npt.NDArray[np.float64] = np.argsort(alphaPos).astype(np.float64)
        betaOrder: npt.NDArray[np.float64] = np.argsort(betaPos).astype(np.float64)
        deltaOrder: npt.NDArray[np.float64] = np.argsort(deltaPos).astype(np.float64)
        
        r1: npt.NDArray[np.float64] = np.random.rand(nVar)
        r2: npt.NDArray[np.float64] = np.random.rand(nVar)
        
        A: npt.NDArray[np.float64] = 2 * a * r1 - a
        C: npt.NDArray[np.float64] = 2 * r2
        
        D_alpha: npt.NDArray[np.float64] = np.abs(C * alphaOrder - currentOrder)
        D_beta: npt.NDArray[np.float64] = np.abs(C * betaOrder - currentOrder)
        D_delta: npt.NDArray[np.float64] = np.abs(C * deltaOrder - currentOrder)
        
        X_alpha: npt.NDArray[np.float64] = alphaOrder - A * D_alpha
        X_beta: npt.NDArray[np.float64] = betaOrder - A * D_beta
        X_delta: npt.NDArray[np.float64] = deltaOrder - A * D_delta
        
        newOrder: npt.NDArray[np.float64] = (X_alpha + X_beta + X_delta) / np.float64(3.0)
        
        newPos: npt.NDArray[np.int64] = self.orderToPermutation(newOrder, nVar)
        
        return newPos

    def isValidPermutation(self,
                          position: npt.NDArray[np.int64]) -> bool:
        return len(np.unique(position)) == len(position) and bool(np.all(position >= 0)) and bool(np.all(position < len(position)))

    def correctPermutation(self,
                         position: npt.NDArray[np.int64],
                         nVar: int) -> npt.NDArray[np.int64]:
        used: set = set(position)
        missing: list[int] = [i for i in range(nVar) if i not in used]
        newPos: npt.NDArray[np.int64] = position.copy()
        
        seen: set = set()
        for i in range(nVar):
            if newPos[i] in seen and missing:
                newPos[i] = missing.pop(0)
            seen.add(newPos[i])
        
        return newPos

    def orderToPermutation(self,
                          order: npt.NDArray[np.float64],
                          nVar: int) -> npt.NDArray[np.int64]:
        sorted_indices = np.argsort(order)
        permutation: npt.NDArray[np.int64] = np.zeros(nVar, dtype=np.int64)
        for i, idx in enumerate(sorted_indices):
            permutation[idx] = i
        return permutation

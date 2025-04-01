from typing import Any, Callable, Tuple
import numpy as np
import numpy.typing as npt

from SolutionClass import Solution


class TabuSearch:
    def __init__(self,
                 maxIt: int       = 100,
                 nNeighbors: int  = 10,
                 tabuTenure: int  = 5,
                 maxStagnant: int = 20):
        
        self.maxIt: int       = maxIt
        self.nNeighbors: int  = nNeighbors
        self.tabuTenure: int  = tabuTenure
        self.maxStagnant: int = maxStagnant

    def initialize(self,
                   population: npt.NDArray[Any],
                   costFunction: Callable[[npt.NDArray[Any]], float]) \
    -> Solution:
        
        nPopulations: int = population.size
        bestSol: Solution = Solution()
        for i in range(nPopulations):
            currentSol: Solution = population[i].copy()
            
            tabuList: list[Tuple[int, int]] = []
            stagnantCount: int = 0

            neighbors: npt.NDArray[Any] = np.array([])
            
            for _ in range(self.maxIt):
                neighbors = np.array([Solution() for _ in range(self.nNeighbors)])
                moves: list[Tuple[int, int]] = []
                
                for j in range(self.nNeighbors):
                    newPos, move = self.createNeighbor(currentSol.Position, tabuList)
                    neighbors[j].Position = newPos
                    neighbors[j].Cost = costFunction(newPos)
                    moves.append(move)
                
                bestNeighborIdx: int = -1
                bestNeighborCost: float = np.inf
                
                for j in range(self.nNeighbors):
                    isTabu: bool = moves[j] in tabuList
                    if (not isTabu and neighbors[j].Cost < bestNeighborCost) or \
                       (isTabu and neighbors[j].Cost < bestSol.Cost):
                        bestNeighborIdx = j
                        bestNeighborCost = neighbors[j].Cost
                
                if bestNeighborIdx == -1:
                    break
                    
                currentSol = neighbors[bestNeighborIdx].copy()
                currentMove: Tuple[int, int] = moves[bestNeighborIdx]
                
                tabuList.append(currentMove)
                if len(tabuList) > self.tabuTenure:
                    tabuList.pop(0)
                
                if currentSol.Cost < bestSol.Cost:
                    bestSol = currentSol.copy()
                    stagnantCount = 0
                else:
                    stagnantCount += 1
                
                if stagnantCount >= self.maxStagnant:
                    break
            
            population[i] = bestSol.copy()
        
        return bestSol

    def createNeighbor(self,
                      position: npt.NDArray[np.int64],
                      tabuList: list[Tuple[int, int]]) \
    -> Tuple[npt.NDArray[np.int64], Tuple[int, int]]:
        
        nVar: int = position.size
        newPos: npt.NDArray[np.int64] = position.copy()
        
        maxAttempts: int = 10
        for _ in range(maxAttempts):
            i, j = np.random.choice(nVar, 2, replace=False)
            move: Tuple[int, int] = (min(i, j), max(i, j))
            if move not in tabuList:
                newPos[i], newPos[j] = newPos[j], newPos[i]
                return newPos, move
        
        i, j = np.random.choice(nVar, 2, replace=False)
        newPos[i], newPos[j] = newPos[j], newPos[i]
        move: Tuple[int, int] = (min(i, j), max(i, j))
        return newPos, move

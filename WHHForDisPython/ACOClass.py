from typing import Any, Callable, List
import numpy as np
import numpy.typing as npt

from SolutionClass import Solution


class ACO:
    def __init__(self,
                 model = None,
                 nAnts: int        = 50,
                 nGenerations: int = 100,
                 alpha: float      = 1.0,
                 beta: float       = 1.0,
                 rho: float        = 0.05,
                 Q: float = 1.0):
        
        self.eta: npt.NDArray[np.float64] | None
        self.nAnts: int        = nAnts
        self.nGenerations: int = nGenerations
        self.alpha: float      = alpha
        self.beta: float       = beta
        self.rho: float        = rho
        self.Q: float = Q
        if model:
            self.eta = 1.0 / (np.array(model["d"]) + 1e-10)
        else:
            self.eta = None

    def initialize(self,
                   population: npt.NDArray[Any],
                   costFunction: Callable[[npt.NDArray[Any]], float]) \
    -> Solution:
        
        nVar: int = population[0].Position.size
        bestSol: Solution = population[0].copy()
        
        tau: npt.NDArray[np.float64] = np.ones((nVar, nVar)) * 0.1


        for _ in range(self.nGenerations):
            for i in range(self.nAnts):
                for j in range(nVar - 1):
                    tau[population[i].Position[j], population[i].Position[j+1]] += self.Q / population[i].Cost
                tau[population[i].Position[-1], population[i].Position[0]] += self.Q / population[i].Cost
                
                visited: List[bool] = [False] * nVar
                cur: int = np.random.randint(nVar)
                visited[cur] = True
                position: List[int] = [cur]

                while False in visited:
                    unvisited = np.where(np.logical_not(visited))[0]
                    probs = np.zeros(len(unvisited))

                    for i, unvisited_point in enumerate(unvisited):
                        probs[i] = tau[cur, unvisited_point] ** self.alpha
                        if self.eta is not None:
                            probs[i] *= self.eta[cur, unvisited_point] ** self.beta

                    probs /= probs.sum()
                    next_point = np.random.choice(unvisited, p=probs)
                    position.append(next_point)
                    visited[next_point] = True
                    cur = next_point

                population[i] = Solution(Position=np.array(position),
                                         Cost=costFunction(np.array(position)))
                if population[i].Cost < bestSol.Cost:
                    bestSol = population[i].copy()
            tau *= self.rho

        population.sort()

        return bestSol

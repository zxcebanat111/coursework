from typing import Any, Callable, Tuple
import numpy as np
import numpy.typing as npt

from SolutionClass import Solution
from RouletteWheelSelection import RouletteWheelSelection


class GA:

    def __init__(self,
                 nPopulations: int = 50,
                 nGenerations: int = 100,
                 pc: float = 0.8,
                 pm: float = 0.2):

        self.nPopulations = nPopulations
        self.nGenerations = nGenerations
        self.pc = pc
        self.pm = pm
        self.nCross = 2 * round(nPopulations * pc / 2)
        self.nMut = round(nPopulations * pm)
        

    def initialize(self,
                   population: npt.NDArray[Any],
                   costFunction: Callable[[npt.NDArray[Any]], float]) \
    -> Solution:
        
        nPopulations: int = population.size
        bestSol: Solution = Solution()
        for _ in range(self.nGenerations):
            crossPopulation: npt.NDArray[Any] = np.array([Solution() for _ in range(self.nCross)])
            self.Crossover(crossPopulation,
                           population)

            for i in range(self.nCross):
                crossPopulation[i].Cost = costFunction(crossPopulation[i].Position)

            mutPopulation: npt.NDArray[Any] = np.array([Solution() for _ in range(self.nMut)])
            self.Mutation(mutPopulation,
                          population)

            newPopulations: npt.NDArray[Any] = np.sort(np.concatenate([population,
                                                                       crossPopulation,
                                                                       mutPopulation]))
            for i in range(nPopulations):
                population[i] = newPopulations[i]

            bestSol = newPopulations[0].copy()
        return bestSol


    def Crossover(self,
                  crossPopulation: npt.NDArray[Any],
                  population: npt.NDArray[Any]) -> None:
        nVar: int = population[0].Position.size
        
        f: npt.NDArray[np.float64] = np.array([1.0 / (sol.Cost + 1e-10) for sol in population])
        f = f / f.sum()
        
        for k in range(0, self.nCross, 2):
            parent1_idx: int = RouletteWheelSelection(f)
            parent2_idx: int = RouletteWheelSelection(f)
            while parent2_idx == parent1_idx:
                parent2_idx = RouletteWheelSelection(f)
            
            parent1: Solution = population[parent1_idx]
            parent2: Solution = population[parent2_idx]
            
            start, end = sorted(np.random.choice(nVar, 2, replace=False))
            
            child1 = np.full(nVar, -1, dtype=np.int64)
            child2 = np.full(nVar, -1, dtype=np.int64)
            
            child1[start:end] = parent1.Position[start:end]
            child2[start:end] = parent2.Position[start:end]
            
            def fill_child(child: npt.NDArray[np.int64], source: npt.NDArray[np.int64]) -> None:
                source_idx = 0
                child_idx = 0
                while child_idx < nVar:
                    if child_idx == start:
                        child_idx = end
                    if child_idx < nVar and child[child_idx] == -1:
                        while source[source_idx] in child:
                            source_idx += 1
                        child[child_idx] = source[source_idx]
                        child_idx += 1
                    else:
                        child_idx += 1
            
            fill_child(child1, parent2.Position)
            fill_child(child2, parent1.Position)
            
            crossPopulation[k].Position = child1
            crossPopulation[k + 1].Position = child2 if k + 1 < self.nCross else crossPopulation[k].Position

    def Mutation(self,
                 mutPopulation: npt.NDArray[Any],
                 population: npt.NDArray[Any]):

        def Swap(x: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
            nVar = x.size
            i, j = np.random.choice(np.arange(nVar), 2)
            xCopy = x.copy()
            xCopy[i], xCopy[j] = xCopy[j], xCopy[i]
            return xCopy

        for i in range(self.nMut):
            j = np.random.randint(self.nPopulations)
            mutPopulation[i].Position = Swap(population[j].Position)


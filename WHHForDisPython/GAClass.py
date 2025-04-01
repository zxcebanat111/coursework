from typing import Any, Callable, Tuple
import numpy as np
import numpy.typing as npt

from SolutionClass import Solution


class GA:

    def __init__(self,
                 nPopulations: int,
                 pc: float = 0.8,
                 pm: float = 0.2):

        self.nPopulations = nPopulations
        self.pc = pc
        self.pm = pm
        self.nCross = 2 * round(nPopulations * pc / 2)
        self.nMut = round(nPopulations * pm)
        

    def initialize(self,
                   population: npt.NDArray[Any],
                   costFunction: Callable[[npt.NDArray[Any]], float]) \
    -> Solution:
        
        nPopulations: int = population.size
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

        bestSol: Solution = newPopulations[0].copy()
        return bestSol


    def Crossover(self,
                  crossPopulation: npt.NDArray[Any],
                  population: npt.NDArray[Any]):

        f: npt.NDArray[np.float32] = np.array([sol.Cost for sol in population])
        f = 1.0 / (f + 1e-10)
        f /= f.sum() + 1e-10
        f = np.cumsum(f)

        for k in range(0, self.nCross - 1, 2):
            i: int = int(np.searchsorted(f, np.random.rand(), 'left'))
            j: int = int(np.searchsorted(f, np.random.rand(), 'left'))

            crossPopulation[k].Position = population[i].Position.copy()
            crossPopulation[k + 1].Position = population[j].Position.copy()


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


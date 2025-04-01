from typing import Any, Callable
import numpy as np
import numpy.typing as npt

from RouletteWheelSelection import RouletteWheelSelection
from SolutionClass import Solution


class ICA:
    def __init__(self,
                 nCountries: int  = 50,
                 nImp: int       = 10,
                 maxIt: int      = 50,
                 beta: float     = 2.0,
                 pRev: float     = 0.1):
        
        self.nCountries: int = nCountries
        self.nImp: int      = nImp
        self.maxIt: int     = maxIt
        self.beta: float    = beta
        self.pRev: float    = pRev

    def initialize(self,
                   population: npt.NDArray[Any],
                   costFunction: Callable[[npt.NDArray[Any]], float]) \
    -> Solution:
        
        nPopulations: int = population.size
        nVar: int = population[0].Position.size
        bestSol: Solution = Solution()
        
        countries: npt.NDArray[Any] = population.copy()
        for i in range(nPopulations):
            if not self.isValidPermutation(countries[i].Position):
                countries[i].Position = np.random.permutation(nVar)
            countries[i].Cost = costFunction(countries[i].Position)
        
        countries = np.sort(countries)
        
        imperialists: npt.NDArray[Any] = countries[:self.nImp].copy()
        colonies: npt.NDArray[Any] = countries[self.nImp:].copy()
        
        nColonies: int = len(colonies)
        impPower: npt.NDArray[np.float64] = np.array([1.0 / (imp.Cost + 1e-10) 
                                                    for imp in imperialists])
        impPower = impPower / impPower.sum()
        empireSizes: npt.NDArray[np.int64] = np.round(impPower * nColonies).astype(int)
        
        empires: list[list[Solution]] = [[] for _ in range(self.nImp)]
        colony_idx: int = 0
        for i in range(self.nImp):
            nCol: int = empireSizes[i]
            for _ in range(nCol):
                if colony_idx < nColonies:
                    empires[i].append(colonies[colony_idx].copy())
                    colony_idx += 1
        
        for _ in range(self.maxIt):
            for i in range(self.nImp):
                imp: Solution = imperialists[i]
                for col in empires[i]:
                    newPos: npt.NDArray[np.int64] = self.assimilate(col.Position, imp.Position)
                    col.Position = newPos
                    col.Cost = costFunction(col.Position)
                    
                    if np.random.rand() < self.pRev:
                        col.Position = self.revolution(col.Position)
                        col.Cost = costFunction(col.Position)
                    
                    if col.Cost < imp.Cost:
                        imperialists[i], empires[i][empires[i].index(col)] = col.copy(), imp.copy()
                
                all_sols: list[Solution] = [imperialists[i]] + empires[i]
                empire_best: Solution = min(all_sols, key=lambda x: x.Cost)
                if empire_best.Cost < bestSol.Cost:
                    bestSol = empire_best.copy()
            
            if len(imperialists) > 1:
                totalCosts: npt.NDArray[np.float64] = np.zeros(self.nImp)
                empire_costs: list[float] = []
                for i in range(self.nImp):
                    empire_costs: list[float] = [c.Cost for c in empires[i]]
                    totalCosts[i] = imperialists[i].Cost + (0.1 * np.mean(empire_costs) if empire_costs else 0)
                
                powers: npt.NDArray[np.float64] = np.float64(1.0) / (totalCosts + 1e-10)
                powers = powers / powers.sum()
                
                weakest_idx: int = int(np.argmax(totalCosts))
                if len(empire_costs) > 0:
                    winner_idx: int = RouletteWheelSelection(powers)
                    if winner_idx != weakest_idx:
                        transferred: Solution = empires[weakest_idx].pop(0)
                        empires[winner_idx].append(transferred)
                
                i: int = 0
                while i < len(imperialists):
                    if not empires[i]:
                        imperialists = np.delete(imperialists, i)
                        empires.pop(i)
                        self.nImp -= 1
                    else:
                        i += 1
        
        all_solutions: list[Solution] = []
        for i in range(self.nImp):
            all_solutions.append(imperialists[i])
            all_solutions.extend(empires[i])
        sorted_sols: npt.NDArray[Any] = np.sort(np.array(all_solutions))
        for i in range(min(nPopulations, len(sorted_sols))):
            population[i] = sorted_sols[i].copy()
        
        return bestSol

    def assimilate(self,
                   colonyPos: npt.NDArray[np.int64],
                   impPos: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        
        nVar: int = len(colonyPos)
        newPos: npt.NDArray[np.int64] = colonyPos.copy()
        
        nSwaps: int = max(1, int(self.beta * np.random.rand() * nVar / 2))
        
        for _ in range(nSwaps):
            diff_idx: npt.NDArray[np.int64] = np.where(newPos != impPos)[0]
            if len(diff_idx) < 2:
                break
            
            i, j = np.random.choice(diff_idx, 2, replace=False)
            newPos[i], newPos[j] = newPos[j], newPos[i]
        
        return newPos

    def revolution(self,
                   position: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        
        nVar: int = len(position)
        newPos: npt.NDArray[np.int64] = position.copy()
        
        i, j = np.random.choice(nVar, 2, replace=False)
        newPos[i], newPos[j] = newPos[j], newPos[i]
        
        return newPos

    def isValidPermutation(self,
                          position: npt.NDArray[np.int64]) -> bool:
        return len(np.unique(position)) == len(position) and \
               bool(np.all(position >= 0)) and \
               bool(np.all(position < len(position)))

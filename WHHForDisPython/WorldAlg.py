from RouletteWheelSelection import RouletteWheelSelection
from SolutionClass import Solution

from typing import Any, Dict, List
import numpy as np
import numpy.typing as npt
import tqdm


def RunWHH(N: int,
           nPopulation: int,
           MaxIt: int,
           algorithms: Dict[str, Any],
           RLAlpha: float,
           CostFunction):

    nAlg = len(algorithms)

    population = np.array([Solution() for _ in range(nPopulation)])

    BestSol: Solution = population[0]

    for i in range(nPopulation):

        population[i].Position = np.random.permutation(N)

        population[i].Cost = CostFunction(population[i].Position)

        if population[i].Cost <= BestSol.Cost:
            BestSol = population[i].copy()

    BestCostMaxiter = np.ones(MaxIt + 1) * np.inf
    BestSolMaxiter: List[Solution] = [Solution() for _ in range(MaxIt + 1)]
    names = list(algorithms.keys())
    algRewards = {name : -np.inf for name in names}
    savedPopulation = population.copy()
    solutions: Dict[str, Solution] = {}
    for name in names:
        pop = savedPopulation.copy()
        solutions[name] = algorithms[name].initialize(pop, CostFunction)
        algRewards[name] = 1.0 / (solutions[name].Cost + 1e-10)
        algorithms['sa'].T *= algorithms['sa'].alpha
        if solutions[name].Cost < BestSol.Cost:
            BestSol = solutions[name].copy()
    BestCostMaxiter[0] = BestSol.Cost
    BestSolMaxiter[0] = BestSol.copy()
    improvements_to_del = {name : {"count" : 0, "sum" : 0.0} for name in names}

    rewards = list(sorted(list(algRewards.items()), key=lambda x : x[1]))

    pop = savedPopulation.copy()
    for iter in range(1, MaxIt + 1):
        print("------------------------------")
        print("Iteration", iter)
        chRand = np.random.rand()
        if chRand > RLAlpha:
            algInd = np.random.randint(nAlg)
            lotCH = names[algInd]
        else:
            newValues = np.array([item[1] for item in rewards])
            P: npt.NDArray[np.float64] = newValues / newValues.sum()
            lotCH = rewards[RouletteWheelSelection(P)][0]
        solutions[lotCH] = algorithms[lotCH].initialize(pop, CostFunction)
        # print(f"Chosen Algorithm : {lotCH}")
        # print(f"Current Best Found Cost : {BestSol.Cost}")
        # print(f"Best Found Cost by Algorithm: {solutions[lotCH].Cost}")
        if solutions[lotCH].Cost < BestSol.Cost:
            improvements_to_del[lotCH]["count"] += 1
            improvements_to_del[lotCH]["sum"] += BestSol.Cost - solutions[lotCH].Cost
            BestSol = solutions[lotCH].copy()

        BestCostMaxiter[iter] = BestSol.Cost
        BestSolMaxiter[iter] = BestSol.copy()
        algRewards[lotCH] = algRewards[lotCH] + BestCostMaxiter[iter - 1] - BestCostMaxiter[iter]
        if BestCostMaxiter[iter - 1] == BestCostMaxiter[iter]:
            algRewards[lotCH] -= float(np.mean(list(algRewards.values())))

        RLAlpha += 1 / MaxIt
        rewards = list(sorted(list(algRewards.items()), key=lambda x : x[1]))
        pop.sort()
        # print("Iteration:", iter, "Best Cost:", BestCostMaxiter[iter])
        algorithms['sa'].T *= algorithms['sa'].alpha
        print('Improvements by algorithms')
        for item in improvements_to_del.items():
            print(f"{item[0]} : {item[1]}")
        print('AlgRewards by algorithms')
        for item in algRewards.items():
            print(f"{item[0]} : {item[1]}")
    return BestCostMaxiter, BestSolMaxiter

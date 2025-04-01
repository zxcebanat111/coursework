from ACOClass import ACO
from CreateModel import CreateModel
from GAClass import GA
from GWOClass import GWO
from ICAClass import ICA
from RouletteWheelSelection import RouletteWheelSelection
from SAClass import SA
from SolutionClass import Solution
from TabuClass import TabuSearch
from TourLength import TourLength

from functools import partial
import numpy as np
import numpy.typing as npt
import tqdm


def RunWHH(x, y):
# Insert Data
    model = CreateModel(x, y)
    CostFunction = partial(TourLength, distances=model["d"])

# Parameter settings
    nPopulation = 50
    MaxIt = 50
    nAlg = 6
    RLAlpha = 0


# GA parameters
    GAObject = GA(nPopulation)

# SA parameters
    SAObject = SA()

# ACO parameters
    ACOObject = ACO()

# ICA parameters
    ICAObject = ICA()

# Taboo parameters
    TabuObject = TabuSearch()

# GWO parameters
    GWOObject = GWO()

# Initialization

# Create empty structure for individuals

# Create Population Array
    population = np.array([Solution() for _ in range(nPopulation)])

# Initialize Best Solution
    BestSol = Solution()

# Initialize Population
    for i in range(nPopulation):

        # Initialize Position
        population[i].Position = np.random.permutation(model["n"])

        # Evaluation
        population[i].Cost = CostFunction(population[i].Position)

        # Update Best Solution
        if population[i].Cost <= BestSol.Cost:
            BestSol = population[i].copy()

# Array to hold best cost values
    BestCostMaxiter = np.ones(MaxIt) * np.inf
    names = ['sa', 'ga', 'aco', 'tabu', 'ica', 'gwo']
    algRewards = {name : -np.inf for name in names}

    savedPopulation = population.copy()

    pop = savedPopulation.copy()

    BestSolSA: Solution = SAObject.initialize(pop, CostFunction)
    algRewards['sa'] = 1.0 / (BestSolSA.Cost + 1e-10)
    SAObject.T *= SAObject.alpha


    pop = savedPopulation.copy()

    BestSolGA: Solution = GAObject.initialize(pop, CostFunction)
    algRewards['ga'] = 1.0 / (BestSolGA.Cost + 1e-10)
    SAObject.T *= SAObject.alpha


    pop = savedPopulation.copy()

    BestSolACO: Solution = ACOObject.initialize(pop, CostFunction)
    algRewards['aco'] = 1.0 / (BestSolACO.Cost + 1e-10)
    SAObject.T *= SAObject.alpha


    pop = savedPopulation.copy()

    BestSolTabu: Solution = TabuObject.initialize(pop, CostFunction)
    algRewards['tabu'] = 1.0 / (BestSolTabu.Cost + 1e-10)
    SAObject.T *= SAObject.alpha


    pop = savedPopulation.copy()

    BestSolICA: Solution = ICAObject.initialize(pop, CostFunction)
    algRewards['ica'] = 1.0 / (BestSolICA.Cost + 1e-10)
    SAObject.T *= SAObject.alpha


    pop = savedPopulation.copy()

    BestSolGWO: Solution = GWOObject.initialize(pop, CostFunction)
    algRewards['gwo'] = 1.0 / (BestSolGWO.Cost + 1e-10)
    SAObject.T *= SAObject.alpha

    rewards = list(sorted(list(algRewards.items()), key=lambda x : x[1]))
    for iter in tqdm.tqdm(range(MaxIt)):
        chRand = np.random.rand()
        if chRand > RLAlpha:
            algInd = np.random.randint(nAlg)
            lotCH = names[algInd]
        else:
            rewValues = np.array([item[1] for item in rewards])
            P: npt.NDArray[np.float64] = rewValues / rewValues.sum()
            lotCH = rewards[RouletteWheelSelection(P)][0]
        if lotCH == 'sa':
            BestSolSA: Solution = SAObject.initialize(pop, CostFunction)
            algRewards[lotCH] = 1.0 / (BestSolSA.Cost + 1e-10)

            if BestSolSA.Cost < BestSol.Cost:
                BestSol = BestSolSA.copy()
        elif lotCH == 'ga':
            BestSolGA: Solution = GAObject.initialize(pop, CostFunction)
            algRewards[lotCH] = 1.0 / (BestSolGA.Cost + 1e-10)

            if BestSolGA.Cost < BestSol.Cost:
                BestSol = BestSolGA.copy()
        elif lotCH == 'aco':
            BestSolACO: Solution = ACOObject.initialize(pop, CostFunction)
            algRewards[lotCH] = 1.0 / (BestSolACO.Cost + 1e-10)

            if BestSolACO.Cost < BestSol.Cost:
                BestSol = BestSolACO.copy()
        elif lotCH == 'tabu':
            BestSolTabu: Solution = TabuObject.initialize(pop, CostFunction)
            algRewards[lotCH] = 1.0 / (BestSolTabu.Cost + 1e-10)

            if BestSolTabu.Cost < BestSol.Cost:
                BestSol = BestSolTabu.copy()
        elif lotCH == 'ica':
            BestSolICA: Solution = ICAObject.initialize(pop, CostFunction)
            algRewards[lotCH] = 1.0 / (BestSolICA.Cost + 1e-10)

            if BestSolICA.Cost < BestSol.Cost:
                BestSol = BestSolICA.copy()
        elif lotCH == 'gwo':
            BestSolGWO: Solution = GWOObject.initialize(pop, CostFunction)
            algRewards[lotCH] = 1.0 / (BestSolGWO.Cost + 1e-10)

            if BestSolGWO.Cost < BestSol.Cost:
                BestSol = BestSolGWO.copy()
        BestCostMaxiter[iter] = BestSol.Cost
        if iter > 1:
            algRewards[lotCH] = algRewards[lotCH] + BestCostMaxiter[iter - 1] - BestCostMaxiter[iter]
            if BestCostMaxiter[iter - 1] == BestCostMaxiter[iter]:
                algRewards[lotCH] -= float(np.mean(list(algRewards.values())))
        else:
            algRewards[lotCH] = algRewards[lotCH] + BestCostMaxiter[iter]

        RLAlpha += 1 / MaxIt
        rewards = list(sorted(list(algRewards.items()), key=lambda x : x[1]))
        # print("Iteration:", iter, "Best Cost:", BestCostMaxiter[iter])
        SAObject.T *= SAObject.alpha
    return BestCostMaxiter

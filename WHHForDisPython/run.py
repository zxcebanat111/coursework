from functools import partial
from TourLength import TourLength
from WorldAlg import RunWHH
from CreateModel import CreateModel

from ACOClass import ACO
from GAClass import GA
from GWOClass import GWO
from ICAClass import ICA
from SAClass import SA
from TabuClass import TabuSearch

import numpy as np
import matplotlib.pyplot as plt

N = 100 
nPopulation = 50
MaxIt = 50
Algorithms = {"sa" : SA(),
             "ga" : GA(),
             "tabu" : TabuSearch(),
             "ica" : ICA(),
             "gwo" : GWO()}
RLAlpha = 0.0
cost = TourLength

xs = []
ys = []

plt.figure(figsize=(12, 8))
xgrid = np.linspace(0, MaxIt, MaxIt)
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'purple', 'orange', 'pink']
iterations = []
solutions = []
for i in range(10):
    print(f"Run #{i+1}")
    x = np.random.randint(0, 100, N)
    y = np.random.randint(0, 100, N)
    model = CreateModel(x, y)
    Algorithms['aco'] = ACO(model)
    CostFunction = partial(cost, distances=model["d"])
    xs.append(x.copy())
    ys.append(y.copy())
    WHHCosts, WHHSols = RunWHH(N,
                               nPopulation,
                               MaxIt,
                               Algorithms,
                               RLAlpha,
                               CostFunction)
    plt.plot(xgrid, WHHCosts[1:], 
             color=colors[i], 
             label=f'WHH Run n. {i}',
             linewidth=1.5)
    iterations.append(WHHCosts[1:])
    solutions.append(WHHSols[-1].Position)


plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(np.min(iterations)*0.9, np.max(iterations)*1.1)
plt.margins(x=0.05)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Found minimum', fontsize=12)
plt.title('10 WHH Runs on different data', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('/home/bebro/coursework/WHHForDisPython/graphics/10Runs.png', 
            dpi=300,
            bbox_inches='tight',
            format='png')
plt.show()


plt.figure(figsize=(12, 8))
xgrid = np.linspace(0, 50, 50)
means = np.mean(iterations, axis=0)
plt.plot(xgrid, means,
         label="Mean minimum found in 10 runs",
         linewidth=1.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(np.min(means)*0.9, np.max(means)*1.1)
plt.margins(x=0.05)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Average minimum', fontsize=12)
plt.title('Average minimum found by iteration', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('/home/bebro/coursework/WHHForDisPython/graphics/MeanRuns.png', 
            dpi=300,
            bbox_inches='tight',
            format='png')
plt.show()


for i in range(10):
    plt.figure(figsize=(12, 8))
    xgrid = np.linspace(0, 100, 50)
    ygrid = np.linspace(0, 100, 50)
    plt.scatter(xs[i], ys[i])
    connected_x = [xs[i][j] for j in solutions[i]]
    connected_y = [ys[i][j] for j in solutions[i]]
    plt.plot(connected_x, connected_y)
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.margins(x=0.05)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Found path', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1))
    # plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    # plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f'/home/bebro/coursework/WHHForDisPython/graphics/Path{i}.png', 
                dpi=300,
                bbox_inches='tight',
                format='png')
    plt.show()


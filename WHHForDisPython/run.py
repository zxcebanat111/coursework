from WorldAlg import RunWHH

import numpy as np
import matplotlib.pyplot as plt

N = 10 
plt.figure(figsize=(12, 8))
xgrid = np.linspace(0, 50, 50)
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'purple', 'orange', 'pink']
iterations = []
for i in range(10):
    x = np.random.randint(0, 100, N)
    y = np.random.randint(0, 100, N)
    WHHCosts = RunWHH(x, y)
    plt.plot(xgrid, WHHCosts, 
             color=colors[i], 
             label=f'WHH Run n. {i}',
             linewidth=1.5)
    iterations.append(WHHCosts)

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
plt.savefig('tex/graphics/10Runs.png', 
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
plt.savefig('tex/graphics/MeanRuns.png', 
            dpi=300,
            bbox_inches='tight',
            format='png')
plt.show()

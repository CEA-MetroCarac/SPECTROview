import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

fig = Figure()
ax = fig.add_subplot(111)
ax.plot([1, 2], [3, 4])

plt.style.use("dark_background")

fig.patch.set_facecolor(plt.rcParams['figure.facecolor'])
fig.patch.set_edgecolor(plt.rcParams['figure.edgecolor'])
ax.set_facecolor(plt.rcParams['axes.facecolor'])
for spine in ax.spines.values():
    spine.set_color(plt.rcParams['axes.edgecolor'])
ax.tick_params(colors=plt.rcParams['xtick.color'], which='both')
ax.xaxis.label.set_color(plt.rcParams['axes.labelcolor'])
ax.yaxis.label.set_color(plt.rcParams['axes.labelcolor'])
ax.title.set_color(plt.rcParams['axes.labelcolor'])

print("dark ax facecolor:", ax.get_facecolor())
print("dark fig facecolor:", fig.get_facecolor())
print("dark spine color:", ax.spines['bottom'].get_color())

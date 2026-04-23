import matplotlib.pyplot as plt
from matplotlib.figure import Figure

fig = Figure()
ax = fig.add_subplot(111)

plt.style.use("dark_background")
ax.clear()
print("dark ax facecolor after clear:", ax.get_facecolor())
print("dark fig facecolor after clear:", fig.get_facecolor())

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
try:
    from qbstyles import mpl_style
    mpl_style(dark=True)
    print("qbstyles dark applied.")
except Exception as e:
    print("Error:", e)

print("axes.facecolor:", plt.rcParams['axes.facecolor'])
print("figure.facecolor:", plt.rcParams['figure.facecolor'])

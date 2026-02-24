
import json
import numpy as np
import matplotlib.pyplot as plt

filename = "GRP25_2022B1A81559G_2022A8PS0671G.json"

with open(filename, "r") as f:
    data = json.load(f)

weights = np.array(data["weights"][0]) 

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle("Weight Visualization of the 20 Hidden Neurons (Diverging Color Scheme)", fontsize=16)

v_abs_max = np.max(np.abs(weights))

for i, ax in enumerate(axes.flat):
    reshaped_weights = weights[i].reshape(28, 28)

    im = ax.imshow(reshaped_weights, cmap='RdBu', vmin=-v_abs_max, vmax=v_abs_max) 
    
    ax.set_title(f"Neuron {i}")
    ax.axis('off')

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Weight Value (Red: Positive, Blue: Negative)')

plt.subplots_adjust(top=0.9, right=0.9)
plt.show()

fig.savefig("GRP25_WEIGHT_VISUALIZATION.png")
print("Visualization saved as GRP25_WEIGHT_VISUALIZATION.png")
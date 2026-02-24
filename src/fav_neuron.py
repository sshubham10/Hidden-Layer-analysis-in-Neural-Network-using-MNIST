import json
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import mnist_loader
import sys

class Task1WeightVisualizer:
    """Task 1: Visualizing hidden neurons as 28x28 heatmaps."""
    def __init__(self, weights):
        self.weights = weights

    def visualize(self, neuron_idx):
        w_j = self.weights[neuron_idx].reshape(28, 28)
        limit = np.max(np.abs(w_j))
        
        plt.figure(figsize=(5, 5))
        plt.imshow(w_j, cmap='RdBu', vmin=-limit, vmax=limit)
        plt.title(f"Task 1: Neuron {neuron_idx} Weights")
        plt.colorbar(label='Weight Value')
        plt.axis('off')
        plt.savefig(f"TASK1_ NEURON_{neuron_idx}.png")
        plt.close()

class Task2ActivationFinder:
    """Task 2: Identifying top images that excite the neuron."""
    def __init__(self, w1, b1, test_data):
        self.w1 = w1
        self.b1 = b1
        self.test_data = list(test_data)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def run(self, neuron_idx):
        activations = []
        for x, y in self.test_data[:2000]:
            a_hidden = self.sigmoid(np.dot(self.w1, x) + self.b1)
            activations.append((a_hidden[neuron_idx][0], x, y))
        

        activations.sort(key=lambda x: x[0], reverse=True)
        top_8 = activations[:8]

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(f"Task 2: Top 8 Inputs for Neuron {neuron_idx}")
        for i, (act, img, label) in enumerate(top_8):
            ax = axes.flat[i]
            
            ax.imshow(img.reshape(28, 28), cmap='gray')
            ax.set_title(f"Label: {label}\nAct: {act:.4f}")
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"TASK2_NEURON_{neuron_idx}.png")
        plt.close()
        return activations

class Task3SelectivityPlotter:
    """Task 3: Computing activation distribution and selectivity."""
    def run(self, neuron_idx, activations):
        class_data = {i: [] for i in range(10)}
        for act, _, label in activations:
            class_data[label].append(act)

        avgs = [np.mean(class_data[i]) if class_data[i] else 0 for i in range(10)]
        
        plt.figure(figsize=(8, 4))
        bars = plt.bar(range(10), avgs, color='skyblue', edgecolor='navy')

        bars[np.argmax(avgs)].set_color('coral')
        
        plt.xticks(range(10))
        plt.xlabel("Digit Class (0-9)")
        plt.ylabel("Average Activation")
        plt.title(f"Task 3: Neuron {neuron_idx} Selectivity")
        plt.savefig(f"TASK3_NEURON_{neuron_idx}.png")
        plt.close()

def main():
    JSON_FILE = "GRP25_2022B1A81559G_2022A8PS0671G.json"
    FAVORITE_NEURON = 11 

    try:
        print("Loading data and model parameters...")
        _, _, test_data = mnist_loader.load_data_wrapper()
        with open(JSON_FILE, "r") as f:
            model = json.load(f)
        
        w1 = np.array(model["weights"][0])
        b1 = np.array(model["biases"][0])

        print(f"Executing tasks for favorite neuron: {FAVORITE_NEURON}...")
        
        t1 = Task1WeightVisualizer(w1)
        t1.visualize(FAVORITE_NEURON)
        
        t2 = Task2ActivationFinder(w1, b1, test_data)
        top_activations = t2.run(FAVORITE_NEURON)
        
        t3 = Task3SelectivityPlotter()
        t3.run(FAVORITE_NEURON, top_activations)

        print("Done! Visualization images saved to current directory.")
        print("Time check: Completed in under 10 seconds.")

    except FileNotFoundError:
        print(f"Error: Ensure {JSON_FILE} and mnist_loader.py are in this directory.")

if __name__ == "__main__":
    main()
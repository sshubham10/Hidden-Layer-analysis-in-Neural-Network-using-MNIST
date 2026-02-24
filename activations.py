import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mnist_loader

training_data, _, _ = mnist_loader.load_data_wrapper()
data_subset = list(training_data)[:5000] 
filename = "train.json"

with open(filename, "r") as f:
    model = json.load(f)

w1 = np.array(model["weights"][0]) 
b1 = np.array(model["biases"][0])

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

print("Calculating activations for all neurons...")
all_activations = {j: {i: [] for i in range(10)} for j in range(20)}

for x, y in data_subset:
    a_hidden = sigmoid(np.dot(w1, x) + b1)
    label = np.argmax(y)
    for j in range(20):
        all_activations[j][label].append(a_hidden[j][0])

print("Generating 20 selectivity plots...")
for j in range(20):
    avg_activations = [np.mean(all_activations[j][i]) if all_activations[j][i] else 0 for i in range(10)]
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(10), avg_activations, color='skyblue', edgecolor='navy')
    
    max_idx = np.argmax(avg_activations)
    bars[max_idx].set_color('coral')
    
    plt.xticks(range(10))
    plt.xlabel("Digit Class (0-9)")
    plt.ylabel("Average Activation")
    plt.title(f"Class-Selectivity for Neuron {j}")
    
    plt.savefig(f"NEURON_{j}_SELECTIVITY.png")
    plt.close()

print("Done! Check your folder for 20 PNG files.")

import json
import numpy as np
import matplotlib

matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import mnist_loader

print("Loading data and model...")
training_data, _, _ = mnist_loader.load_data_wrapper()
training_list = list(training_data) 

with open("train.json", "r") as f:
    model = json.load(f)

w1 = np.array(model["weights"][0])
b1 = np.array(model["biases"][0])

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

for target_neuron in range(20):
    print(f"Processing Neuron {target_neuron}...")
    activations = []

    for x, y in training_list[:5000]:
        a_hidden = sigmoid(np.dot(w1, x) + b1)
        activations.append((a_hidden[target_neuron][0], x, np.argmax(y)))

    activations.sort(key=lambda x: x[0], reverse=True)
    top_8 = activations[:8]


    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f"Task 2: Top 8 Inputs for Neuron {target_neuron}")

    for i, (act, img, label) in enumerate(top_8):
        ax = axes.flat[i]
        ax.imshow(img.reshape(28, 28), cmap='gray')
        ax.set_title(f"Label: {label}\nAct: {act:.4f}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"Neuron_{target_neuron}_Top8.png")
    plt.close()

print("All 20 neuron grids have been saved as PNG files in your folder.")

## **MNIST Neural Network: Internal Feature Interpretability**

This repository explores the internal representations of a feedforward neural network trained on the MNIST handwritten digit dataset. Rather than treating the model as a "black box," this project focuses on visualizing what individual hidden layer neurons actually "see" and how they categorize visual primitives to achieve classification.

---

### **Overview**

The core engine utilizes a three-layer architecture $[784, 20, 10]$ implemented using Stochastic Gradient Descent and Backpropagation. The project transitions from basic training to a deep-dive analysis of hidden layer weights and activation profiles.

### **Technical Implementation**

*  **Optimal Training Configuration**: Achieved a peak test accuracy of **~94.1%** using a learning rate of **3.0**, a mini-batch size of **10**, and **30 epochs**.


*  **Stochasticity**: Used smaller batch sizes to introduce beneficial noise, helping the model avoid local minima.


*  **Interpretation Engine**: Custom scripts were developed to extract weight vectors and reshape them into $28\times28$ heatmaps to analyze excitatory (positive) and inhibitory (negative) regions.



---

### **Key Insights: Feature vs. Class Selectivity**

One of the primary findings of this project is that the hidden layer does not learn in a uniform way. Instead, it develops two distinct types of "experts":

#### **1. Feature-Selective Neurons**

These neurons act as "building block detectors." They fire based on specific geometric strokes or curves shared across many digits.

*  **Neuron 0 ("Upper Curve Detector")**: Triggers on curved strokes shared by '0' and '2'.


*  **Neuron 13 ("Loop Detector")**: Shows generalized activation across almost all classes, acting as a background or edge detector.



#### **2. Class-Selective Neurons**

These neurons act as specialized "digit detectors" that have a dominant preference for a single category.

*  **Neuron 8 ("Digit 4 Detector")**: Specifically tuned to identify the parallel vertical strokes of a '4'.


*  **Neuron 19 ("Digit 0 Detector")**: Almost exclusively activates for the circular structure of a '0'.



---

### **Project Structure**

```
├── src/
│   ├── training_network.py   # Hyperparameter tuning and training logic
│   ├── network.py            # Core engine (Backprop & SGD) [Nielsen]
│   ├── mnist_loader.py       # Data loading utility [Nielsen]
│   ├── visualize_weights.py  # Heatmap generation for Task 1
│   ├── activations.py        # Top-activating image extraction for Task 2
│   ├── analyze_neurons.py    # Class-selectivity and distribution for Task 3
│   └── fav_neuron.py         # Main script to showcase the "Favorite Neuron"
├── model/
│   └── train.json    # Saved parameters (94%+ Accuracy)
├── report/
│   └── report.pdf# Detailed analysis report
├── visuals/
│   └── Detailed visualization using jpg and png images for each neuron
└── README.md


```

### **How to Run**

To reproduce the analysis for the favorite hidden neuron (**Neuron 11 - "Jalebi"** ), run:

```bash
python fav_neuron.py

```

The script loads weights from the JSON file and generates the weight heatmap, top-8 inputs, and selectivity bar chart in under 10 seconds.

---

### **Credits & Attribution**

*  **Base Implementation**: Core neural network engine adapted from *Neural Networks and Deep Learning* by Michael Nielsen. https://github.com/mnielsen/neural-networks-and-deep-learning


*  **Analysis & Visualization**: All technical implementations of Tasks 1-3, including the weight heatmaps, activation studies, and interpretability conclusions, were developed independently.



---

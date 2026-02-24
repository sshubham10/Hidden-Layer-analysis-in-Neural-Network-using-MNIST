import mnist_loader
import network
import json

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 20, 10])

net.SGD(training_data, 30, 10, 3, test_data=test_data)

filename = "GRP25_2022B1A81559G_2022A8PS0671G.json"
net.save(filename)
print(f"Model saved as {filename}")
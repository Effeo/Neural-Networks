
from typing import List
import pandas as pd
import numpy as np

from library.activations import ReLU, Softmax
from library.conv_dense import ConvDense
from library.dense import Dense
from library.layer import Layer
from library.network import predict, train
from library.reshape import Reshape
from library.utils import cross_entropy_loss, derivatie_cross_entropy, get_accuracy, get_predictions

def make_predictions(x: np.ndarray, network: List[Layer]) -> np.ndarray:
    
    outputs = []
    for image in x:
        o = predict(network, image)
        outputs.append(o)
    
    outputs = np.array(outputs)
    
    return get_predictions(outputs)
    

data = pd.read_csv(r'.\\train.csv')
data = np.array(data)

# m = 42000
# n = 784 + 1 per le label

m, n = data.shape
np.random.shuffle(data)

data_dev = data[0: 1000].T

# Ogni colonna è un immagine
# la prima riga sono le label

Y_dev = data_dev[0] # sono le label
X_dev = data_dev[1: n] # sono le immagini 
X_dev = X_dev / 255. # Normalizza i pixel tra [0, 1]

print(f"m = {m} - n = {n}")
print(f"data_dev = {data_dev.shape}")
print(f"X_dev = {X_dev.shape}")
print(f"Y_dev = {Y_dev.shape}")

data_train = data[1000: m].T
Y_train = data_train[0] # sono le label
X_train = data_train[1:n] # sono le immagini
X_train = X_train / 255. # Normalizziamo i pixel tra [0, 1]
_,m_train = X_train.shape

print(f"X_train: {X_train.shape}")
print(f"Y_train: {Y_train.shape}")

network_8K = [
    Dense(784, 10),
    ReLU(),
    Softmax(True),
]

errors_8k, accuracies_8K = train(
    x_train=X_train,
    y_train=Y_train,
    epochs=500,
    learning_rate=0.1,
    loss_prime=derivatie_cross_entropy,
    loss_function=cross_entropy_loss,
    network=network_8K,
    use_r_prop=False
)

dev_predictions = make_predictions(X_dev, network_8K)
print("Test accuracy:")
test_accuracy_network_8K = get_accuracy(dev_predictions, Y_dev)
print(test_accuracy_network_8K)
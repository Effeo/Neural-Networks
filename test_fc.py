from keras.datasets import mnist
import numpy as np
from library.activations import ReLU, Sigmoid, Softmax
from library.conv_dense import ConvDense
from library.dense import Dense

from library.layer import Layer
from library.network import predict, train
from library.utils import cross_entropy_loss, derivatie_cross_entropy, flatten_list
from keras.utils import to_categorical

def preprocess_data(x: np.ndarray, y: np.ndarray, limit: int = 60000):
    unique_classes = np.unique(y)  # Find the unique classes in y
    all_indices = []  # To store selected indices for all classes

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0][:limit]  # Get indices for class up to limit
        all_indices.append(cls_indices)
    
    all_indices = np.hstack(all_indices)  # Combine indices from all classes
    all_indices = np.random.permutation(all_indices)  # Shuffle the combined indices

    x = x[all_indices]  # Select balanced set of features
    y = y[all_indices]  # Select balanced set of labels
    
    x_train = flatten_list(x)
    x_train = np.array(x_train)
    x_train = x_train.astype("float32") / 255
    print(x_train.shape) # (60000, 169, 16)
    y_train = to_categorical(y, num_classes=10)  # One-hot encode y
    y_train = y_train.reshape(len(y), 10, 1)
    
    print(y_train.shape) # (60000, 10, 1)
    return x_train, y_train

network = [
    ConvDense((169, 16), 1), # ConvDense. K = 16x1 
    ReLU(),
    Dense(169, 100),
    Sigmoid(),
    Dense(100, 10),
    Softmax(),
]


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

train(
    x_train=x_train,
    y_train=y_train,
    epochs=40,
    learning_rate=0.01,
    loss_prime=derivatie_cross_entropy,
    loss_function=cross_entropy_loss,
    network=network,
    use_r_prop=False
)

correct = 0
c = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    if np.argmax(output) == np.argmax(y):
        correct = correct + 1
    c = c+1

print('Result: ' + str(correct) + '/' + str(c))

# print(x_train.shape) # (60000, 28, 28)
# print(x_train[0].shape) # (28, 28)
# print(x_test.shape) # (10000, 28, 28)


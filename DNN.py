# coding: utf-8

# # Deep Neural Network For Image Classification From Scratch

# In[ ]:

# function to load the training and test set
import os
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt


# =====================================
def load_data(training_set, test_set):
    """
    Arguments:
    training_set -- folder containing training set images (image name == label)
    test_set -- folder containng test set images (image name == label)

    Returns:
    training_set_features -- 2d array of size (nx, m)
    training_set_labels -- 2d array of size (1, m)
    test_set_features -- 2d array of size (nx, m)
    test_set_labels -- 2d array of size (1, m)
    """

    # load the training set data
    infiles = os.listdir(training_set)
    training_set_features = []
    training_set_labels = []
    training_set_features = []
    training_set_labels = []

    for image in infiles:
        label = image.split(".")[0]
        if label == "cat":
            training_set_labels.append(0)
        else:
            training_set_labels.append(1)
        im = plt.imread(training_set + "/" + image)
        shape = im.shape
        features = im.reshape(shape[0] * shape[1] * shape[2])
        training_set_features.append(features)

    training_set_features = np.asarray(training_set_features, dtype=float)
    training_set_features = training_set_features.T
    training_set_labels = np.asarray(training_set_labels, dtype=float)
    training_set_labels = training_set_labels.reshape(1, len(training_set_labels))

    # Load the test set data
    infiles = os.listdir(test_set)
    test_set_features = []
    test_set_labels = []
    test_set_features = []
    test_set_labels = []

    for image in infiles:
        label = image.split(".")[0]
        if label == "cat":
            test_set_labels.append(0)
        else:
            test_set_labels.append(1)
        im = plt.imread(test_set + "/" + image)
        shape = im.shape
        features = im.reshape(shape[0] * shape[1] * shape[2])
        test_set_features.append(features)

    test_set_features = np.asarray(test_set_features, dtype=float)
    test_set_features = test_set_features.T
    test_set_labels = np.asarray(test_set_labels, dtype=float)
    test_set_labels = test_set_labels.reshape(1, len(test_set_labels))

    return training_set_features, training_set_labels, test_set_features, test_set_labels


# In[ ]:

def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


# In[ ]:

# load the data
train_f, train_l, test_f, test_l = load_data("train_set2", "test_set")


# In[ ]:

def initialize_parameters(n_x, n_h1, n_h2, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h1 -- size of the 1st hidden layer
    n_h2 -- size of the 2nd hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h1, n_x)
                    b1 -- bias vector of shape (n_h1, 1)
                    W2 -- weight matrix of shape (n_h2, n_h1)
                    b2 -- bias vector of shape (n_h2, 1)
                    W3 -- weight matrix of shape (n_y, n_h2)
                    b3 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) * 0.01
    b3 = np.zeros((n_y, 1))

    ### END CODE HERE ###

    assert (W1.shape == (n_h1, n_x))
    assert (b1.shape == (n_h1, 1))
    assert (W2.shape == (n_h2, n_h1))
    assert (b2.shape == (n_h2, 1))
    assert (W3.shape == (n_y, n_h2))
    assert (b3.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3
                  }

    return parameters


# In[ ]:

parameters = initialize_parameters(30000, 10, 5, 1)
print(parameters["W1"].shape, "W1")
print(parameters["b1"].shape, "b1")
print(parameters["W2"].shape, "W2")
print(parameters["b2"].shape, "b2")
print(parameters["W3"].shape, "W3")
print(parameters["b3"].shape, "b3")


# In[ ]:

def sigmoid(z, takederivative):
    s = 1 / (1 + np.exp(-z))
    if takederivative == "derivative":
        return s * (1 - s)
    else:
        return s


# In[ ]:

def forward_propogation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2", "A2", and "Z3" , "A3
    """
    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
    A1 = sigmoid(Z1, "notderivative")

    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2 = sigmoid(Z2, "notderivative")

    Z3 = np.dot(parameters["W3"], A2) + parameters["b3"]
    A3 = sigmoid(Z3, "notderivative")

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3}

    return A3, cache


# In[ ]:

parameters = initialize_parameters(30000, 10, 5, 1)
A3, cache = forward_propogation(train_f, parameters)
print(A3.shape, "A2")
print(cache["Z1"].shape, "Z1")
print(cache["A1"].shape, "A1")
print(cache["Z2"].shape, "Z2")
print(cache["A2"].shape, "A2")
print(cache["Z3"].shape, "Z3")
print(cache["A3"].shape, "A3")


# In[ ]:

def compute_cost(A3, Y):
    """
    Computes the cross-entropy cost

    Arguments:
    A3 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    m = A3.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(A3).T) + np.dot(1 - Y, np.log(1 - A3).T));
    return cost


# In[ ]:

compute_cost(A3, train_l)


# In[ ]:

def back_propogation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2", "A2" and Z3 and A3.
    X -- input data of shape (nx, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with
    """
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    m = X.shape[1]

    dZ3 = A3 - Y
    dW3 = 1 / m * (np.dot(dZ3, A2.T))
    db3 = 1 / m * (np.sum(dZ3, axis=1, keepdims=True))

    dZ2 = np.dot(W3.T, dZ3) * sigmoid(cache["Z2"], "derivative")
    dW2 = 1 / m * (np.dot(dZ2, A1.T))
    db2 = 1 / m * (np.sum(dZ2, axis=1, keepdims=True))

    dZ1 = np.dot(W2.T, dZ2) * sigmoid(cache["Z1"], "derivative")
    dW1 = 1 / m * (np.dot(dZ1, X.T))
    db1 = 1 / m * (np.sum(dZ1, axis=1, keepdims=True))

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dW3": dW3,
             "db3": db3}

    return grads


# In[ ]:

grads = back_propogation(parameters, cache, train_f, train_l)


# In[ ]:

# GRADED FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate=0.01):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    ### END CODE HERE ###

    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]
    ## END CODE HERE ###

    # Update rule for each parameter
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = W1 - (learning_rate * dW1)
    b1 = b1 - (learning_rate * db1)
    W2 = W2 - (learning_rate * dW2)
    b2 = b2 - (learning_rate * db2)
    W3 = W3 - (learning_rate * dW3)
    b3 = b3 - (learning_rate * db3)
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


# In[ ]:

update_parameters(parameters, grads, learning_rate=0.02)


# In[ ]:

# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h1, n_h2, num_iterations=10000, print_cost=False, learning_rate=0.0075):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    ### START CODE HERE ### (≈ 5 lines of code)
    parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    ### END CODE HERE ###

    # Loop (gradient descent)

    for i in range(0, num_iterations):
        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A3, cache = forward_propogation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A3, Y)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = back_propogation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)

        ### END CODE HERE ###

        # Print the cost every 1000 iterations
        print("Cost after iteration %i: %f" % (i, cost))

    return parameters


# In[ ]:

import h5py

train_dataset = h5py.File('train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # train set labels
train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten / 255.

X, Y = load_planar_dataset()
parameters = nn_model(train_set_x, train_set_y, 50, 10, num_iterations=10000, print_cost=True, learning_rate=0.5)
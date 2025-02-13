# main.py

import ast
import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data
import pandas as pd
import pickle

nnfs.init()
np.random.seed(0)

#print("Python:", sys.version)
#print("Numpy:", np.version)
#print("Matplotlib:", matplotlib._version)




# hidden layers
# from 'Neural Networks from Scratch' (forward functions)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # backpropagation
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
     # backpropagation
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    # backpropagation
    def backward(self, dvalues, y_true):
        samples = len (dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        # Update weights with gradient descent
        layer.weights -= self.learning_rate * layer.dweights
        # Update biases with gradient descent
        layer.biases -= self.learning_rate * layer.dbiases


# inputs
# Load the preprocessed data
train_df = pd.read_csv('train_preprocessed.csv')

# Convert the string representations of lists in 'vectorized_tokens' to actual lists
train_df['vectorized_tokens'] = train_df['vectorized_tokens'].apply(ast.literal_eval)

#print(train_df['vectorized_tokens'])

# Convert the relevant columns to a numpy array
X_train = np.array(train_df['vectorized_tokens'].tolist())
y_train = np.array(train_df['sentiment_encoded'])

#print(X_train)
#print (y_train)

# Check the shape of X_train to ensure it is 2-dimensional
print(X_train.shape) #output (11202, 15)

# Confirm that X_train is a 2D array with the following
if len(X_train.shape) == 2:
    input_size = X_train.shape[1]  # This should match the length of your vectorized data
    output_size = 4
else:
    raise ValueError('The input data is not properly vectorized as a 2-dimensional array.')

#print(input_size)
#print(output_size)


# Initialise layers with the correct sizes
dense1 = Layer_Dense(input_size, 5)  # Choose number of neurons in layer (5)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(5, output_size)  # The output size should match number of classes (4)
activation2 = Activation_Softmax()

# Initialise loss object
loss_function = Loss_CategoricalCrossEntropy()

# Initialise optimizer
optimizer = Optimizer_SGD(learning_rate=0.01)


# Training loop
for epoch in range(1000):  # Number of epochs
    # Pass the training data through the layers
    dense1.forward(X_train)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate loss
   
    loss = loss_function.calculate(activation2.output, y_train)

    # Backpropagation
    loss_function.backward(activation2.output, y_train)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)

    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    # Print loss every 100 epochs (as an example)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}") 


# Print first 5 predictions
print(activation2.output[:5])
        
    

model_parameters = {
    "dense1_weights": dense1.weights,
    "dense1_biases": dense1.biases,
    "dense2_weights": dense2.weights,
    "dense2_biases": dense2.biases
}

# Save model parameters to a file
with open("model_parameters.pkl", "wb") as file:
    pickle.dump(model_parameters, file)




'''
# raw python code of the 'output'
layer_outputs = [] # Output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # Output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
'''
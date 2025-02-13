# training_model.py

import ast
import os
import numpy as np
import pandas as pd
import pickle
import nnfs
from model_definition import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossEntropy, Optimizer_SGD

nnfs.init()
np.random.seed(0)

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['vectorized_tokens'] = df['vectorized_tokens'].apply(ast.literal_eval)
    X = np.array(df['vectorized_tokens'].tolist())
    y = np.array(df['sentiment_encoded'])
    return X, y

# training loop
def train_model(X_train, y_train, epochs=1000, learning_rate=0.001):
    input_size = X_train.shape[1]
    output_size = 4  

     # First layer
    dense1 = Layer_Dense(input_size, 10)
    activation1 = Activation_ReLU()

    # Second layer
    dense2 = Layer_Dense(10, 10)  
    activation2 = Activation_ReLU()

    # Output layer
    dense3 = Layer_Dense(10, output_size)
    activation3 = Activation_Softmax()

    loss_function = Loss_CategoricalCrossEntropy()
    optimizer = Optimizer_SGD(learning_rate)

    for epoch in range(epochs):
        # Forward pass through the first layer
        dense1.forward(X_train)
        activation1.forward(dense1.output)

        # Forward pass through the second layer
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        # Forward pass through the output layer
        dense3.forward(activation2.output)
        activation3.forward(dense3.output)

        # Calculate loss
        loss = loss_function.calculate(activation3.output, y_train)

        # Backpropagation
        loss_function.backward(activation3.output, y_train)
        activation3.backward(loss_function.dinputs)
        dense3.backward(activation3.dinputs)

        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)

        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)

        # Print the loss every 100 epochs (as an example)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}") 
   
    # Print the first 5 predictions
    print("First 5 predictions:", activation3.output[:5])

    return {
        "dense1_weights": dense1.weights, "dense1_biases": dense1.biases,
        "dense2_weights": dense2.weights, "dense2_biases": dense2.biases,
        "dense3_weights": dense3.weights, "dense3_biases": dense3.biases  # Save the new layer's parameters
    }



# saves the model to be used later
def save_model(model_parameters, filepath="model_parameters.pkl"):
    with open(filepath, "wb") as file:
        pickle.dump(model_parameters, file)

# loads the model if exists
def load_model(filepath="model_parameters.pkl"):
    with open(filepath, "rb") as file:
        return pickle.load(file)

# either trains the model or loads existing model
def train_or_load_model(train_filepath, model_filepath="model_parameters.pkl"):
    if os.path.exists(model_filepath):
        print("Loading existing model...")
        model_parameters = load_model(model_filepath)
    else:
        print("Training new model...")
        X_train, y_train = load_data(train_filepath)
        model_parameters = train_model(X_train, y_train)
        save_model(model_parameters, model_filepath)
    return model_parameters

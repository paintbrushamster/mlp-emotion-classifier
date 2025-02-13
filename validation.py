# validation

import numpy as np
import pandas as pd
import ast
from model_definition import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossEntropy

def load_validation_data(filepath):
    df = pd.read_csv(filepath)
    df['vectorized_tokens'] = df['vectorized_tokens'].apply(ast.literal_eval)
    X_val = np.array(df['vectorized_tokens'].tolist())
    y_val = np.array(df['sentiment_encoded'])
    return X_val, y_val


def validate_model(model_parameters, X_val, y_val):
    input_size = X_val.shape[1]  
    output_size = 4  

    # First Layer
    dense1 = Layer_Dense(input_size, 10)
    activation1 = Activation_ReLU()

    # Second layer
    dense2 = Layer_Dense(10, 10)
    activation2 = Activation_ReLU()

    # Output layer
    dense3 = Layer_Dense(10, output_size)
    activation3 = Activation_Softmax()

    # Load weights and biases into layers
    dense1.weights, dense1.biases = model_parameters["dense1_weights"], model_parameters["dense1_biases"]
    dense2.weights, dense2.biases = model_parameters["dense2_weights"], model_parameters["dense2_biases"]
    dense3.weights, dense3.biases = model_parameters["dense3_weights"], model_parameters["dense3_biases"]

    loss_function = Loss_CategoricalCrossEntropy()

    # Forward pass
    dense1.forward(X_val)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    # Get predictions
    predictions = np.argmax(activation3.output, axis=1)

    # Print the first 5 predictions
    print("First 5 validation predictions:", predictions[:5])

    # Calculate loss
    loss = loss_function.calculate(activation3.output, y_val)
    print(f"Validation Loss: {loss}")

    # Calculate accuracy
    if y_val.shape == predictions.shape:  # Ensuring the shapes match
        accuracy = np.mean(predictions == y_val)
        print(f"Validation Accuracy: {accuracy}")
    else:
        print("Error: Mismatched shapes between predicted labels and true labels.")

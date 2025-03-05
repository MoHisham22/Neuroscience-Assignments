import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([[0.05, 0.10]])
weights_input_hidden = np.array([[0.15, 0.20], [0.25, 0.30]])
bias_hidden = np.array([[0.35, 0.35]])
weights_hidden_output = np.array([[0.40, 0.45], [0.50, 0.55]])
bias_output = np.array([[0.60, 0.60]])
target = np.array([[0.01, 0.99]])

hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)
final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
output_output = sigmoid(final_input)

error_output = target - output_output
output_delta = error_output * sigmoid_derivative(output_output)
hidden_delta = np.dot(output_delta, weights_hidden_output.T) * sigmoid_derivative(hidden_output)

lr = 0.5
weights_hidden_output += lr * np.dot(hidden_output.T, output_delta)
bias_output += lr * output_delta
weights_input_hidden += lr * np.dot(inputs.T, hidden_delta)
bias_hidden += lr * hidden_delta

print("Output:", output_output)
print("Weights Input-Hidden:", weights_input_hidden)
print("Bias Hidden:", bias_hidden)
print("Weights Hidden-Output:", weights_hidden_output)
print("Bias Output:", bias_output)

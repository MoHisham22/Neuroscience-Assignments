import numpy as np

def tanh(x):
    return np.tanh(x)
w1 = np.random.uniform(-0.5, 0.5, (2, 2))  
w2 = np.random.uniform(-0.5, 0.5, (2, 1))

b1 = np.array([0.5, 0.5])  
b2 = np.array([0.7])

x = np.array([0.3, -0.2])

hidden_input = np.dot(x, w1) + b1
hidden_output = tanh(hidden_input)

output_input = np.dot(hidden_output, w2) + b2
output = tanh(output_input)

print("Hidden layer output:", hidden_output)
print("Final output:", output)
import numpy as np

# create our input
X = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])
# print("Input:\n", X) 

# create output array 
y = np.array([[1], [1], [0]])
# print("Actual output:\n", y)

# define our sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# derivative of sigmoid function
def d_sigmoid(x):
    return x * (1 - x)

# initialize required variables
epoch = 5000
learning_rate = 0.1
inputlayer_neurons = X.shape[1]
hiddenlayer_neurons = 3
output_neurons = 1

# initializing weight and bias
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# train out model
for i in range(epoch):

    # forward propogation
    hidden_layer_input1 = np.dot(X, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)

    # backpropagation
    E = y - output
    slope_output_layer = d_sigmoid(output)
    slope_hidden_layer = d_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) * learning_rate
    bout += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hiddenlayer) * learning_rate
    bh += np.sum(d_hiddenlayer, axis = 0, keepdims=True) * learning_rate


# This output represents our predicted values for out input vectors
print(output)

# prints [ [0.97916599], [0.96839672], [[0.03978868]] ]

import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL.ImageOps as ImageOps

# The input layer has 784 nodes
# The output layer has 10 nodes
# 3 layers: Input layer, hidden layer and softmax output layer
# number of neurons in hidden layer
num_neurons = 300
learning_rate = 0.4
regularization = 0.01
epochs = 15
batch_size = 100
## layers = 3 fixed
layers = 3
num_node = [784,num_neurons,10]
num_output_node = 10


print('number of neurons in hidden layer = ', num_neurons)
print('learning rate = ', learning_rate)
print('regularization = ', regularization)
print('number of epochs = ', epochs)
print('batch size = ', batch_size)

## Read MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

Ntrain = len(mnist.train.labels)
Ntest = len(mnist.test.labels)

# Sigmoid Function
def sigmoid(X):
    return 1/(1 + np.exp(-X))
def sigmoid_derivate(y):
    return y * (1.0 - y)
# Forward Propagation
def forward_prop(W,X):
	## input layer
	input = np.insert(X,0,1) # adding bias
	z = np.dot(input,W[0]) # multiplying with weights
	hidden_layer_output = sigmoid(z)
	
	## hidden layer
	input = hidden_layer_output
	input = np.insert(input,0,1) # adding bias
	z = np.dot(input,W[1])  # multiplying with weights
	output = np.exp(z)
	output = output/np.sum(output)
	return  output, hidden_layer_output
	

def back_prop(y, W, X, Lambda, mini_batch_size):
    delta = np.empty(2, dtype = object)
    
    delta[0] = np.zeros((785,num_neurons))
    delta[1] = np.zeros((num_neurons + 1,10))
    
    for i in range(mini_batch_size):
        A = np.empty(layers-1, dtype = object)
        A[0] = X[i]
        output, A[1] = forward_prop(W,X[i])
        diff = output - y[i]
        delta[1] = delta[1] + np.outer(np.insert(A[1],0,1),diff)
        diff = np.multiply(np.dot(np.array([W[1][k+1] for k in range(num_node[1])]), diff), sigmoid_derivate(A[1])) 
        delta[0] = delta[0] + np.outer(np.insert(A[0],0,1),diff)
        #print('delta[0] ',delta[0].shape)
    D = np.empty(layers-1, dtype = object)
    for l in range(layers - 1):
        D[l] = np.zeros((num_node[l]+1,num_node[l+1]))
    for l in range(layers-1):
        for i in range(num_node[l]+1):
            if i == 0:
                for j in range(num_node[l+1]):
                    D[l][i][j] = 1/mini_batch_size * delta[l][i][j]
            else:
                for j in range(num_node[l+1]):
                    D[l][i][j] = 1/mini_batch_size * (delta[l][i][j] + Lambda * W[l][i][j]) #Regularization
                    #print('del values = ',D[l][i][j])
    #print('D = ',D)
    return D


def train_nn_model(y, X, learning_rate, iterations, Lambda):

    W = np.empty(layers-1, dtype = object)
    W[0] = np.zeros((785,num_neurons))
    W[1] = np.zeros((num_neurons + 1,10))
    for k in range(iterations):
        print('GD iteration = ',k)
        D = back_prop(y, W, X, Lambda)
        for l in range(layers-1):
            W[l] = W[l] - learning_rate * D[l]
        #print('change in weights = ',np.sum(learning_rate * D[l]))
    return W

def train_nn_model_batch(learning_rate, iterations, Lambda, mini_batch_size):
    steps = int(len(mnist.train.images) / mini_batch_size)
    num_epochs = iterations
    W = np.empty(layers-1, dtype = object)
    #W[0] = np.zeros((785,num_neurons))  ## never initialize weights at zeros
    #W[1] = np.zeros((num_neurons + 1,10))  ## never initialize weights at zeros
    W[0] = np.random.rand(785,num_neurons)/num_neurons
    W[1] = np.random.rand(num_neurons + 1,10)/num_neurons
    for epoch in range(num_epochs):
            print('epoch number = ',epoch+1)
            for i in range(steps):
                lower_bound = i * mini_batch_size
                upper_bound = min((i + 1) * mini_batch_size, len(mnist.train.images))
                X = mnist.train.images[lower_bound:upper_bound, :]
                y = mnist.train.labels[lower_bound: upper_bound, :]
                D = back_prop(y, W, X, Lambda, mini_batch_size)
                for l in range(layers-1):
                    W[l] = W[l] - learning_rate * D[l]
                #print('change in weights = ',np.sum(learning_rate * D[l]))
    return W

train_images = mnist.train.images
train_labels = mnist.train.labels

#finalweights = train_nn_model(train_labels, train_images, 0.1, 10, 0.1)
finalweights = train_nn_model_batch(learning_rate, epochs, regularization, batch_size)

def accuracy(images,labels):
	num_images = len(labels)
	count = 0
	for i in range(num_images):
		H,h = forward_prop(finalweights,images[i])
		#print(H)
		for j in range(num_output_node):
			if H[j] == np.amax(H) and labels[i][j] == 1:
				count = count + 1
	return (count/num_images)

## accuracy testing on train and test mnist data
test_images_mnist = mnist.test.images
test_labels_mnist = mnist.test.labels

# Determine the accuracy of the training data
train_accuracy = accuracy(train_images,train_labels)

# Determine the accuracy of the test data
test_accuracy = accuracy(test_images_mnist, test_labels_mnist)

# USPS data testing
usps_data_count = 1500
usps_per_digit_data = 150
per_digit_label_counter = 0
usps_test_images = np.ndarray([usps_data_count + 1, 784])
image_label = 0
usps_test_labels = np.ndarray([usps_data_count + 1, 10])
required_size = (28, 28)

np.random.seed(0)

def normalize(data):
    row_sums = data.sum(axis=1)
    norm_matrix = np.divide(data, row_sums[:, np.newaxis]) #  increase the dimension of the existing array by one more dimension
    normalized_data = 1 - norm_matrix
    normalized_data[normalized_data < 1] = 0
    return normalized_data

for i in range(usps_data_count, 0, -1):
	file_path = './proj3_images/Test/test_' + "{0:0=4d}".format(i) + '.png'
	img = Image.open(file_path)
	img = img.resize(required_size)

	image = np.asarray(img)

	normalized_data = normalize(image)
	flattened_vector = normalized_data.flatten()
	usps_test_images[i] = flattened_vector

	if (per_digit_label_counter == usps_per_digit_data):
		image_label += 1
		per_digit_label_counter = 0

	per_digit_label_counter += 1
	label = np.zeros(10)
	label[image_label] = 1
	usps_test_labels[i] = label


usps_images = usps_test_images[1:1501]
usps_labels = usps_test_labels[1:1501]

usps_test_accuracy = accuracy(usps_images, usps_labels)

print('mnist train accuracy = ',train_accuracy)
print('mnist test accuracy = ',test_accuracy)
print('usps test accuracy = ',usps_test_accuracy)
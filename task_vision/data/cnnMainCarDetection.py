# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:19:49 2018

@author: SUMAN
"""

#IMPORTING ALL THE NECESSARY PACKAGES
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops

#these are functions written specifically to call up the data and load the important modules accordingly
from cnnfns import convert_to_one_hot , random_mini_batches
from DatagenFinal import LoadData

#loading the car damage dataset
X_train_orig, X_test_orig , Y_train_orig, Y_test_orig = LoadData()

# Example of a car damage dataset
#index = 6
#plt.imshow(X_train_orig[index])
#print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

#determining the size of the dataset before we send the data for training

#Step 1 : Normalising the dataset . As the intensities varies from 0-255 , thus we divide the same by 255
X_train = X_train_orig/255.
X_test = X_test_orig/255.

#converting the labels into one hot encoding as it is a multiclass data
Y_train = convert_to_one_hot(Y_train_orig, 3).T
Y_test = convert_to_one_hot(Y_test_orig, 3).T

#printing the shape of the train and test data
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

#typecasting into floating numbers
X_train= X_train.astype(np.float32)
X_test= X_test.astype(np.float32)
Y_train= Y_train.astype(np.float32)
Y_test= Y_test.astype(np.float32)


#FOLLOWING ARE THE FUNCTIONS REQUIRED TO RUN THE TRAINING DATA AND THEN TEST IT ACCORDINGLY

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape=[None, n_y])
    
    return X, Y


def initialize_parameters():
    
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [5, 5, 3, 6]
                        W2 : [5, 5, 6, 16]
                        W3 : [5 , 5, 16 , 20]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2 , W3
    """
        

    W1 = tf.get_variable('W1',[5, 5, 3, 6], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2',[5, 5, 6, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable('W3',[5, 5, 16, 20], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable('W4',[5, 5, 20, 24], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4}
    
    return parameters

def initialise_parameters_allcnn():
    
    W1 = tf.get_variable("W1" , [3 , 3 , 3 ,96] , initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2" , [3, 3, 96, 96] , initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable("W3" , [3, 3, 96, 192] , initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable("W4" , [3, 3, 192, 192] , initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W5 = tf.get_variable("W5" , [3, 3, 192, 192] , initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W6 = tf.get_variable("W6" , [1, 1, 192, 192] , initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W7 = tf.get_variable("W7" , [1, 1, 192, 10] , initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    parameters = {"W1":W1,
                  "W2":W2,
                  "W3":W3,
                  "W4":W4,
                  "W5":W5,
                  "W6":W6,
                  "W7":W7}
    return parameters

def forward_propagation_allcnn(X , parameters):
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    W6 = parameters['W6']
    W7 = parameters['W7']
    
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    Z2 = tf.nn.conv2d(A1,W2, strides = [1,2,2,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    Z3 = tf.nn.conv2d(A2,W3, strides = [1,1,1,1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)
    Z4 = tf.nn.conv2d(A3,W4, strides = [1,2,2,1], padding = 'SAME')
    A4 = tf.nn.relu(Z4)
    Z5 = tf.nn.conv2d(A4,W5, strides = [1,1,1,1], padding = 'SAME')
    A5 = tf.nn.relu(Z5)
    Z6 = tf.nn.conv2d(A5,W6, strides = [1,1,1,1], padding = 'VALID')
    A6 = tf.nn.relu(Z6)
    Z7 = tf.nn.conv2d(A6,W7, strides = [1,1,1,1], padding = 'VALID')
    A7 = tf.nn.relu(Z7)
    
    P1 = tf.nn.avg_pool(A7 , ksize = [1,6,6,1] , strides = [1,6,6,1] , padding = 'SAME')
    P2 = tf.contrib.layers.flatten(P1)
    
    logitsFinal = tf.contrib.layers.fully_connected(P2, 3 , activation_fn= None)
    
    return logitsFinal

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    
    #4 STACKS OF CONVOLUTIONAL LAYERS
    
    '''layer1 convolutional layer'''
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 2x2, sride 2, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    '''layer2 convolutional layer'''
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    '''layer3 convolutional layer'''
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A3 = tf.nn.relu(Z3)
    # MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P3 = tf.nn.max_pool(A3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    '''layer4 convolutional layer'''
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z4 = tf.nn.conv2d(P3,W4, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A4 = tf.nn.relu(Z4)
    # MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P4 = tf.nn.max_pool(A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    '''flatten out'''
    P5 = tf.contrib.layers.flatten(P4)
    
    '''fully connected layer'''
    F1 = tf.contrib.layers.fully_connected(P5, 120 )
    
    F2 = tf.contrib.layers.fully_connected(F1, 20 )
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    F3 = tf.contrib.layers.fully_connected(F2, 3, activation_fn=None)

    return F3

def compute_cost(Z5, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = Z5, labels = Y)
    cost = tf.reduce_mean(cost)
    
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0005,
          num_epochs = 500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 128, 128, 3)
    Y_train -- test set, of shape (None, n_y = 3)
    X_test -- training set, of shape (None, 128, 128, 3)
    Y_test -- test set, of shape (None, n_y = 3)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    beta = 0.01
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    #parameters = initialise_parameters_allcnn()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    F3 = forward_propagation(X, parameters)
    #F3 = forward_propagation_allcnn(X , parameters)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(F3, Y)
    
    ###ADDING L2 REGULARIZATION###
    #regularizers = tf.nn.l2_loss(parameters['W1']) + tf.nn.l2_loss(parameters['W2'])
    regularizers = tf.nn.l2_loss(parameters['W1']) + tf.nn.l2_loss(parameters['W2']) + tf.nn.l2_loss(parameters['W3']) + tf.nn.l2_loss(parameters['W4'])
    #regularizers = tf.nn.l2_loss(parameters['W1']) + tf.nn.l2_loss(parameters['W2']) + tf.nn.l2_loss(parameters['W3']) + tf.nn.l2_loss(parameters['W4']) + tf.nn.l2_loss(parameters['W5']) + tf.nn.l2_loss(parameters['W6']) + tf.nn.l2_loss(parameters['W7'])
    cost = tf.reduce_mean(cost + beta * regularizers)
    
    # Backpropagation: Define the tensorflow optimizer. AdamOptimizer is used that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(F3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters


TrainingAccuracy , TestingAccuracy , parameters = model(X_train, Y_train, X_test, Y_test)



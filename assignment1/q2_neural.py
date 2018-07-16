#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def output_sigmoid(X, W1, b1):
    """
    Step 1 of forward propogation
    Generate the sigmoid and the sigmoid gradient from input data
    """
    # the X matrix contains a row for each observation (M) and the Dx is the
    # number of dimensions available for x (i.e. how many different words
    # are in the corpus recorded)
    # below we multiply the X-matrix by W1 to get a MxH matrix, where
    xw = np.matmul(X, W1)
    # we want to add the bias for each weights element-wise
    # we then translate each weight for each observation into h, the
    # sigmoid output that will be passed onto the next layer of weights
    # the sigmoid gradient will be used later during back-propagation
    s1 = xw+b1
    print('****s1****')
    print(s1)
    h = sigmoid(s1)
    h_grad = sigmoid_grad(h)
    return h, h_grad

def forward_prop_prediction(h, W2, b2):
    """
    Step 2 of forward propogation
    Generate the prediction yhat from sigmoid and second weights
    """
    # For each output h, we multiply it by another set of weights
    # this will output M x Dy output
    # where Dy representing all the dimension of output word guesses
    # (maps to the one hot vectors)
    # we also want to add the second bias onto each weight
    hw = np.matmul(h, W2)
    s2 = hw + b2
    # the softmax converts every element on Dy into a likelihood
    # and regularizes it so that each row sum up to 1
    # dimension is M x Dy
    yhat = softmax(s2)
    return yhat

def calc_cost_from_prediction(labels, yhat):
    """
    Step 3 of forward propogation
    Calculate cost function of current prediction based on true labels
    To calculate the cost function
    J = sum over i in m ( y_i * log (yhat_i) )
    we multiple the labels M x Dy matrix
    by the log of predicted yhat (M x Dy)
    """
    log_yhat = np.log(yhat)
    y_cost = labels*log_yhat
    j_cost = np.sum(-y_cost)
    return j_cost


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation

    # Generate the prediction using the input and parameters available
    h, h_grad = output_sigmoid(X, W1, b1)
    yhat = forward_prop_prediction(h, W2, b2)
    # print('******Yhat******')
    # print(yhat)
    # calculate cost function
    # print('*****cost*****')
    cost = calc_cost_from_prediction(labels, yhat)
    # print(cost)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    # Derivative of cost relative to z2: dcost / z2 = yhat- y
    # (proven for cost = - sum{ y * log(softmax(z2) }
    # since z2 = h*W2 + b2
    # finding the derivative tells us the direction we need to move
    # against on b2 in order to get cost closer to 0
    # (proved earlier that this is the derivative of softmax)
    # matrix size: M x Dy
    # if we find the average derivative across all observation M
    # then we get the gradient for b2
    print('*****delta1*****')
    delta1 = yhat-labels
    M =delta1.shape[0]
    print(delta1)
    # Derivative of cost relative to h: dcost/ dh = delta1 * w2
    # here we are applying the chain rule to calculate the next derivative
    # delta1 (M x Dy matrix) multiple against the transpose of W2 (Dy, H)
    # the output will be a  M x H matrix
    # but you you need a Dy x M matrix
    print('*****delta2*****')
    delta2 = np.matmul(delta1, np.transpose( W2 ) )
    print(delta2)
    # Derivaive of cost relative to z1
    # we multiply delta2, the derivative of cost relative to h by
    # the derivative of the sigmoid (element-wise multiplication)
    # [TODO] Test delta3, derive this breaking out by element
    print('*****h_grad and h*****')
    print(h)
    print(h_grad)
    print('*****delta3*****')
    delta3 = delta2*h_grad
    print(delta3)

    # Derivative of cost relative to b2
    # print('*****Gradientb2*****')
    one_m = np.ones(M).reshape(1,M)
    # print(one_m)
    gradb2 = np.matmul(one_m, delta1)
    # print(b2)
    # print('-------------')
    # print(gradb2)
    # print('*****GradientW2*****')
    gradW2 = np.matmul(np.transpose(h), delta1)
    # print(W2)
    # print('-------------')
    # print(gradW2)
    # print('*****Gradientb1*****')
    gradb1 = np.matmul(one_m, delta3)
    # print(b1)
    # print('-------------')
    # print(gradb1)
    # print('*****GradientW1*****')
    # [TODO] Test W1 derive this breaking out by element
    gradW1 = np.matmul(np.transpose(X), delta3 )
    print('-----W1--------')
    print(W1)
    print('-----W1-gradient--------')
    print(gradW1)
    print('*****b1*****')
    print(b1)
    print('*****b2*****')
    print(b2)
    print('*****X*****')
    print(X)
    print('*****W2*****')
    print(W2)
    print('*****yhat*****')
    print(yhat)
    print('*****y*****')
    print(labels)

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))
    print(grad)
    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions =  [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()

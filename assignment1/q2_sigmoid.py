#!/usr/bin/env python

import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    denom = (1+np.exp(-x))
    sig = (1/denom)
    ### END YOUR CODE

    return sig


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ### YOUR CODE HERE
    # find the inverse of sigmoid
    s_inv = 1 - s
    # then multiply it element-wise
    # this is not treated as a matrix
    ds = s  * s_inv
    ### END YOUR CODE

    return ds


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    print(f)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "You should verify these results by hand!\n"


def test_sigmoid():
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    x = np.array([
            [0, -4],
            [-2 , 9],
            [-3 , -0.2],
            [-2 ,1],
            [-0.7 ,3]])
    f = sigmoid(x)
    print(f)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array([
        [0.5 , 0.01798621],
        [0.119202922, 0.999876605],
        [0.047425873 , 0.450166003],
        [0.119202922, 0.731058579],
        [0.331812228, 0.952574127]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        [0.25, 0.017662706],
        [0.104993585, 0.000123379],
        [0.04517666, 0.247516573],
        [0.104993585,  0.196611933],
        [0.221712873,  0.04517666]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print('probably right')
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()

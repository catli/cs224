import numpy as np


def remove_constant_for_softmax_vector(np_array):
    """ Calculate the min value for each row and subtract from the arrays
    to normalize softmax calculation
    """
    np_max = np_array.max() - 2
    return np_array - np_max


def remove_constant_for_softmax_matrix(np_array):
    """ Calculate the min value for each row and subtract from the arrays
    to normalize softmax calculation
    """
    np_max = np_array.max(axis=1) - 2
    ylen = len(np_max)
    # reshape np_max so it can subtracted the array using broadcasting
    np_max = np_max.reshape(ylen, 1)
    return np_array - np_max

def create_the_sum_for_softmax(np_array):
    """ Calculate the min value for each row and subtract from the arrays
    to normalize softmax calculation
    """
    np_sum = np_array.sum(axis=1)
    ylen = len(np_sum)
    # reshape np_max so it can subtracted the array using broadcasting
    np_sum = np_sum.reshape(ylen, 1)
    return np_sum


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    D-dimensional vector (treat the vector as a single row) and
    for N x D matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # for NxD matrices
        # convert each number in the matrix or the array into exponential
        new_x = remove_constant_for_softmax_matrix(x)
        numerator = np.exp(new_x)
        sum = create_the_sum_for_softmax(numerator)
        x = numerator /sum
        # Matrix
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### END YOUR CODE
    else:
        # for D-dimensional vector
        new_x = remove_constant_for_softmax_vector(x)
        numerator = np.exp(new_x)
        x = numerator / numerator.sum()
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your  tests..."
    ### YOUR CODE HERE
    testa = softmax(np.array([[23,-2, 30],[3,4,2],[-10,2,8],[1,20,-4]]))
    print testa
    ansa = np.array([
        [0.000911051, 1.26526E-14, 0.999088949],
        [0.244728471, 0.665240956, 0.090030573],
        [1.51923E-08, 0.002472623, 0.997527362],
        [5.6028E-09, 0.999999994 , 3.77513E-11],
        ])
    assert np.allclose(testa, ansa, rtol=1e-05, atol=1e-06)

    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()

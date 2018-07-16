#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
        Given X = [a b c]
    Unit length ||X|| = sqrt(a^2 + b^2 + c^2)
    So in order to normalize the unit length to 1, we find the ||X||
    for each row and then divide each item in the row by that number
    """

    ### YOUR CODE HERE
    # Find the square of each item
    xsq = x*x
    # Find the unit length along each row (axis=1)
    xsq_sum = xsq.sum(axis=1)
    x_unit = np.sqrt(xsq_sum)
    # reshape the unit length and normalize matrix by unit length
    x_unit = x_unit.reshape(x_unit.shape[0],1)
    x = x / x_unit
    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print "Normalize Test Passed"




def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    # identify the target predicted vector and then find the dot product
    # between the vector and the output vectors
    #     outputVector structured as V x D 
    #     v_c structured as 1xD matrix
    # we are assuming here that the  output vector and the 
    # predicted vector is structured so that each row represent a word / token in {1, V}
    v_c = predicted
    z_w = np.dot(outputVectors, v_c)
    # the output yhat is a 1xV matrix
    yhat = softmax(z_w)
    # create the one hot vector for the predicted word
    # calculate the difference for gradient
    ydiff = yhat.copy()
    ydiff[target] -= 1.0

    # find the cross-entropy cost function based on yhat
    # cost = calc_cost_from_prediction(y, yhat)
    cost = - np.log( yhat[target] )

    # calculate the gradient wrt to the v_c (the predicted word vector)
    # the gradient is U(yhat - y)
    # the output should be a D x 1 matrix, same as v_c
    # y is a one-hot vector that represents the actual word
    # and we multiply it by output vector, it can also be calculated
    # by using index to find the vector
    gradPred = np.dot( outputVectors.T, ydiff)


    # calculate the gradient wrt to all other word vectors
    # the gradient is v_c(yhat - y)
    # we multiple yhat by v_c to get a V x D  matrix
    grad = np.outer(ydiff, v_c)

    ### END YOUR CODE
    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    # Similar to softmax, we find the target matrix
    #    v_c structured as 1xD matrix
    #    u_o assume to be 1xD matrix
    #    u_k assume to be K x D matrix
    # we pull the data assuming that each row represent one vector
    v_c = predicted
    u_o =  outputVectors[target]
    u_k = outputVectors[indices]


    # The intermediary matrix outputs
    #    z_o, h_o: single scalar number
    #    z_k, h_k: K x 1 vector, wich each number associated with a neg sample
    z_o = np.dot(u_o, v_c)
    h_o = sigmoid(z_o)
    z_k = np.dot(u_k, v_c)
    h_k = sigmoid( - z_k)

    J_1 = - np.log(h_o)
    J_2 = - np.sum( np.log(h_k) )
    cost = J_1+ J_2

    # Return the gradient for the prediction function
    # the prediction vector interacts with both the predicted vector
    # the negative sample vectors so below are both parts of the gradient
    # here we are trying to increase the prediction matrix to maximize
    # the similarity with the predicted vector
    # output is a 1 x D matrix
    grad_pred_o = - (1 - h_o)*u_o

    # the second part is tyring to decrease
    #  similarity with the negative sample vectors
    # K x 1 multiply be input is a k x D matrix, we will need to sum all negative samples
    # along the rows. output is a 1 x D matrix
    # reshape h_k so that it can multiple
    grad_pred_k = np.dot(( 1 - h_k).T, u_k)
    # find the predicted matrix gradient
    # output is a 1 x D matrix
    gradPred = grad_pred_o + grad_pred_k


    # Return the gradient of the output vector
    # create a matrix the same shape as outputVector
    grad = np.zeros(outputVectors.shape)
    # first find the gradient wrt to the target output vector
    # here we want to increase the similarity between
    # the target output vector and the center vector
    # outputs is a 1 x D matrix
    grad_u_o = - (1-h_o)*v_c

    # print('***************grad_u_o************')
    # print(grad_u_o)
    # print(grad_u_o.shape)
    # replace the target row in output vector gradient
    grad[target, ] = grad_u_o
    # then find the gradient descent of all the u_k matrices
    # K x 1 matrix multiply by 1 x 3
    # K x D
    grad_uk = - np.outer((h_k - 1), v_c)
    # print('***************grad_uk************')
    # print(grad_uk)
    # for each token (row) replace gradient
    for k in xrange(u_k.shape[0]):
        index = indices[k]
        grad[index] += grad_uk[k]

    ### END YOUR CODE
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    ### YOUR CODE HERE

    # Find the position of the context word and extract
    # the  predicted vector that will be the input for skipgram function
    # (1) predicted: extract the predicted vector
    # (the location of the center word)
    # in the inputVector
    currentIndex = tokens[currentWord]
    predicted = inputVectors[currentIndex ]

    # iterate through each target context word and find the cost and gradient
    # the cost and gradient of prediction is the sum of cost and gradient
    # for all context words
    # [TODO] iterate through through contextWords so that we're not duplicating
    # gradient attributed to the context
    # this might mean we need to generate the gradient output by only
    # multiplying by the y_diff values of the target 
    for word in contextWords:
        # target: find the location of each context words
        # from the token dictionary
        target = tokens[word]
        # Input all variables into the function selected
        # both function softMax.. and negSampling.. have the same the same inputs
        # outputVectors: use as is
        word_cost, word_gradPred, word_grad = word2vecCostAndGradient(
            predicted = predicted,
            target = target,
            outputVectors = outputVectors,
            dataset = dataset)

        # add each variable
        cost += word_cost
        # Gradient of v_c the gradient of predicted vector
        # Gradient of v_w for other indices = 0
        gradIn[currentIndex] += word_gradPred
        gradOut += word_grad

    ### END YOUR CODE
    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # since we are tring to predict the center word
    # the target is now the current Word
    target = tokens[currentWord]

    # the predicted vector is the sum of input vectors
    # for all context words
    pred_indices = [tokens[word] for word in contextWords]
    pred_vectors = inputVectors[pred_indices]
    predicted = pred_vectors.sum(axis=0)

    cost, word_gradPred, gradOut = word2vecCostAndGradient( \
            predicted = predicted, \
            target = target, \
            outputVectors = outputVectors, \
            dataset = dataset)

    for i in pred_indices:
        gradIn[i] += word_gradPred

    # raise NotImplementedError
    ### END YOUR CODE
    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        print 'BATCH %f' % i
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()

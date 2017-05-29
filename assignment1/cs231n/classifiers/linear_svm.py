import numpy as np
from random import shuffle
# from past.builtins import xrange

# The key to solve the problem is to figure out how the deviation can be represented.

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape)  # initialize the gradient as zero
  step = 0.00001  # step size for W
  delta = 1  # margin
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)  # scores of different classes of one pic
    correct_class_score = scores[y[i]]  # y[i] gives the right sequence of the class
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + delta  # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i].T  # just a approximation consider that after adding this step the condition(>0 or <0) will not be change
        dW[:, y[i]] -= X[i].T  # just a approximation
  # print (dW.shape)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
   # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  delta = 1  # margin
  dW = np.zeros(W.shape) # initialize the gradient as zero
  step = 0.00001
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_score = np.array([scores[np.array(range(num_train)),y]])
  margin = scores-correct_class_score.T + delta
  margin[np.array(range(num_train)),y] = 0
  # loss = sum(margin[margin>0])- num_train * delta
  loss = sum(margin[margin > 0])
  loss/= num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # pass
  margin_bool = margin > 0
  margin_bool = margin_bool * np.ones(margin_bool.shape)
  margin_bool[np.array(range(num_train)),y] = -(np.sum(margin_bool,axis = 1))
  dW = X.T.dot(margin_bool) / float(num_train)
  dW += 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

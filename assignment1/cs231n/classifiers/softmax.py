import numpy as np
from random import shuffle
# from past.builtins import xrange

# The key to solve the problem is to figure out how the deviation can be represented.



def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  delta_w = 0.00001
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)  # scores of different classes of one pic
    scores = scores-max(scores)
    correct_class_score = scores[y[i]]  # y[i] gives the right sequence of the class
    loss += -np.log(np.exp(correct_class_score)/sum(np.exp(scores)))
    for j in xrange(num_classes):
      dW[:,j] += (np.exp(scores[j])/sum(np.exp(scores))-(j == y[i])) * X[i, :]
  loss = loss / num_train
  dW = dW / num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W    # The coefficient is due to gradient
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the grSadient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  scores = scores-np.reshape(np.array([scores.max(axis=1)]), [num_train, 1])
  correct_class_score = scores[np.array(range(num_train)), y]  # y[i] gives the right sequence of the class
  loss += np.mean(-np.log(np.exp(correct_class_score)/np.sum(np.exp(scores), axis=1)))
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  inner_matrix = np.zeros([num_train, num_classes])
  inner_matrix[range(num_train), y] = 1
  dW =np.dot(X.T,(np.exp(scores) / np.array([np.sum(np.exp(scores),axis=1)]).T - inner_matrix) )
  dW = dW / num_train
  dW += 2*reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    N = X.shape[0]  # numbers of training samples
    C = W.shape[1]  # numbers of classes

    def softmax(a):
        # vector
        a_shift = a - np.max(a)
        s = np.exp(a_shift) / np.sum(np.exp(a_shift))
        return s

    for i in range(N):
        # dW_step = np.zeros_like(W)
        z = X[i].dot(W)
        a = softmax(z)
        loss += -np.log(a)[y[i]]  # negative

        # pay attention to the derivative of softmax function
        # at last is a form similar as "a_i - y_i"
        a[y[i]] -= 1
        for c in range(C):
            dW[:, c] += a[c] * X[i]

        # for c in range(C):
        #     if c == y[i]:
        #         dW_step[:, c] = (a[c] - 1) * X[i]
        #     else:
        #         dW_step[:, c] = a[c] * X[i]
        # dW += dW_step
    
    # num_features, num_classes = W.shape
    # num_train = X.shape[0]
    # for n in range(num_train):
    #     x = X[N]
    #     scores = x.dot(W)
    #     max_score = np.max(scores)
    #     exp_scores = np.exp(scores - max_score)
    #     summedExponentialScores = np.sum(exp_scores)
    #     probs = exp_scores / summedExponentialScores
    #     loss += -np.log(probs[y[n]])

    #     for i in range(num_features):
    #         for j in range(num_classes):
    #             dW[i, j] += (probs[j] - (j == y[n])) * x[i]

    loss = loss / N
    dW = dW / N

    loss = loss + 0.5 * reg * np.sum(W*W)
    dW = dW + reg * W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    N = X.shape[0]

    def softmax(x):
        # matrix, should be familiar with "broadcast" mechanism
        # x_shift = (x.T - np.max(x, axis=1)).T
        x_shift = x - np.max(x, axis=1, keepdims=True)
        s = np.exp(x_shift) / np.sum(np.exp(x_shift), axis=1, keepdims=True)
        return s
    
    z = X.dot(W)
    a = softmax(z)
    # pick out the correspending softmax output
    # a_class = a[-1,y] 
    a_class = a[range(N),y]
    ## should be more skillful at the indexing

    loss = np.sum(-np.log(a_class)) / N
    loss += 0.5 * reg * np.sum(W*W)

    a[range(N),y] -= 1
    # a[-1,y] -= 1 , wrong! don't use -1 index, means find element in the last row
    ## N is where we use accumulation, interpretation of matrix multiplication
    # W_ij = \sum(X_ki \cdot A_kj)
    dW = X.T.dot(a) / N
    dW += reg * W


    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 1e-6]
    # learning_rates = [1e-6, 1e-5] # for 3_features
    # learning_rates = [1e-6]
    regularization_strengths = [1e2, 1e3, 1e4]
    # regularization_strengths = [1e1, 1e2, 1e3, 1e4, 1e5] # for 3_features

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    num_iter = 5000
    for lr in learning_rates:
        for reg in regularization_strengths:
            model = SoftmaxClassifier()
            model.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=num_iter)

            train_accuracy = np.mean(y_train == model.predict(X_train))
            validation_accuracy = np.mean(y_val == model.predict(X_val))

            results[(lr,reg)] = (train_accuracy, validation_accuracy)
            all_classifiers.append((model, validation_accuracy, lr, reg))

            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_softmax = model

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers

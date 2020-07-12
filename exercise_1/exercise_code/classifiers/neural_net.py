"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################

        hidden_layer = np.dot(X, W1) + b1
        hidden_layer_act = np.maximum(hidden_layer, 0)
        # (x + np.abs(x)) / 2.0
        # activation = np.vectorize(lambda x: x * (x>0))
        # hidden_layer_act = activation(hidden_layer)
        output_layer = np.dot(hidden_layer_act, W2) + b2
        scores = output_layer

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################

        # stable edition
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        # scores_exp = np.exp(scores)
        item = scores_exp[list(range(N)),y] / np.sum(scores_exp, axis=1, keepdims=False)
        # scores_exp[list(range(N)),y].shape = (N,)
        # if print(scores_exp[list(range(N)),y]), still in a form of row vector
        # np.sum(scores_exp, axis=1, keepdims=True).shape = (N,1)
        loss = np.sum(-np.log(item)) / N

        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################

        probability = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
        dscore = probability
        dscore[list(range(N)),y] -= 1
        dscore = dscore / N
        # pay attention the the normalization

        dW2 = hidden_layer_act.T.dot(dscore)
        dW2 += reg * W2

        # matrix X W + np.ones(N,1) b.T = Z
        # derivative w.r.t b is Z.T.dot(np.ones(N,1)), equivalent to sum up
        # along axis=0
        db2 = np.sum(dscore, axis=0, keepdims=False)
        # print(db2.shape) = (3,) ; also print((db2.T).shape) = (3,) in toy net
        
        # propagate to output of hidden layer
        dH = dscore.dot(W2.T)
        # before activation
        dH[hidden_layer_act <= 0] = 0

        dW1 = X.T.dot(dH)
        dW1 += reg * W1

        db1 = np.sum(dH, axis=0, keepdims=False)

        # actually i think the transpose is meaningless, since both are numpy
        # array and have the same shape
        grads['W1'] = dW1
        grads['b1'] = db1.T
        grads['W2'] = dW2
        grads['b2'] = db2.T

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1) # exact devision

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################

            index = np.random.choice(num_train, batch_size)
            X_batch = X[index]
            y_batch = y[index]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden_output = np.maximum(np.dot(X,W1)+b1, 0)

        scores = hidden_output.dot(W2) + b2
        
        y_pred = np.argmax(scores, axis=1)        

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 

    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above in the Jupyther Notebook; these visualizations   #
    # will have significant qualitative differences from the ones we saw for   #
    # the poorly tuned network.                                                #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################

    input_size = 32 * 32 * 3
    num_class = 10
    # hidden_size = [64, 128, 256, 512]
    hidden_size = 128
    learning_rates = [1e-3]
    # learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # learning_rate = 1e-3
    regularization = [0.2, 0.5, 1.0]
    # regularization = [0.2, 0.5, 1.0, 2.0, 4.0]
    # regularization = 1.0
    num_iters = 5000 # 10000
    # batch_size = [128, 256, 512]
    batch_size = 256

    results = {}
    best_stats = None
    best_val_acc = -1

    for lr in learning_rates:
      for reg in regularization:
        net = TwoLayerNet(input_size, hidden_size, num_class)
        stats = net.train(X_train, y_train, X_val, y_val,
                  learning_rate=lr, reg=reg, num_iters=num_iters, 
                  batch_size=batch_size)

        # plt.plot(stats['train_acc_history'], label='train')
        # plt.plot(stats['val_acc_history'], label='val')
        # plt.title('Classification accuracy history')
        # plt.xlabel('Epoch')
        # plt.ylabel('Clasification accuracy')
        # plt.legend()

        val_acc = (net.predict(X_val)==y_val).mean()
        print('lr: %f, reg: %f Validation Accuracy: %f' %(lr, reg, val_acc))
        results[(lr, reg)] = stats
        if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_net = net
          best_stats = stats

    # for hs in hidden_size:
    #   for bs in batch_size:
    #     net = TwoLayerNet(input_size, hs, num_class)
    #     stats = net.train(X_train, y_train, X_val, y_val,
    #               learning_rate=learning_rate, reg=regularization, num_iters=num_iters, 
    #               batch_size=bs)

    #     # plt.plot(stats['train_acc_history'], label='train')
    #     # plt.plot(stats['val_acc_history'], label='val')
    #     # plt.title('Classification accuracy history')
    #     # plt.xlabel('Epoch')
    #     # plt.ylabel('Clasification accuracy')
    #     # plt.legend()

    #     val_acc = (net.predict(X_val)==y_val).mean()
    #     print('hs: %f, bs: %f Validation Accuracy: %f' %(hs, bs, val_acc))
    #     results[(hs, bs)] = stats
    #     if val_acc > best_val_acc:
    #       best_val_acc = val_acc
    #       best_net = net
    #       best_stats = stats

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net

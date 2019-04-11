import graph
import coarsening
import utils

import tensorflow as tf
import time, shutil
import numpy as np
import os, collections, sklearn
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx

number_edges = 8
metric = 'euclidean'
normalized_laplacian = True
coarsening_levels = 4
# Directories.
dir_data = 'data_mnist'


# %%
# Here we proceed at computing the original grid where the images live and the various coarsening that are applied
# for each level

def grid_graph(m):
    z = graph.grid(m)  # normalized nodes coordinates
    dist, idx = graph.distance_sklearn_metrics(z, k=number_edges, metric=metric)
    # dist contains the distance of the 8 nearest neighbors for each node indicated in z sorted in ascending order
    # idx contains the indexes of the 8 nearest for each node sorted in ascending order by distance

    A = graph.adjacency(dist, idx)  # graph.adjacency() builds a sparse matrix out of the identified edges computing similarities as: A_{ij} = e^(-dist_{ij}^2/sigma^2)

    return A, z


def coarsen(A, levels):
    graphs, parents = coarsening.metis(A, levels)  # Coarsen a graph multiple times using Graclus variation of the METIS algorithm.
    # Basically, we randomly sort the nodes, we iterate on them and we decided to group each node
    # with the neighbor having highest w_ij * 1/(\sum_k w_ik) + w_ij * 1/(\sum_k w_kj)
    # i.e. highest sum of probabilities to randomly walk from i to j and from j to i.
    # We thus favour strong connections (i.e. the ones with high weight wrt all the others for both nodes)
    # in the choice of the neighbor of each node.

    # Construction is done a priori, so we have one graph for all the samples!

    # graphs = list of spare adjacency matrices (it contains in position
    #          0 the original graph)
    # parents = list of numpy arrays (every array in position i contains
    #           the mapping from graph i to graph i+1, i.e. the idx of
    #           node i in the coarsed graph -> that is, the idx of its cluster)
    perms = coarsening.compute_perm(parents)  # Return a list of indices to reorder the adjacency and data matrices so
    # that two consecutive nodes correspond to neighbors that should be collapsed
    # to produce the coarsed version of the graph.
    # Fake nodes are appended for each node which is not grouped with anybody else
    laplacians = []
    for i, A in enumerate(graphs):
        M, M = A.shape

        # We remove self-connections created by metis.
        A = A.tocoo()
        A.setdiag(0)

        if i < levels:  # if we have to pool the graph
            A = coarsening.perm_adjacency(A, perms[i])  # matrix A is here extended with the fakes nodes
            # in order to do an efficient pooling operation
            # in tensorflow as it was a 1D pooling

        A = A.tocsr()
        A.eliminate_zeros()
        Mnew, Mnew = A.shape
        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added), |E| = {3} edges'.format(i, Mnew, Mnew - M, A.nnz // 2))

        L = graph.laplacian(A, normalized=normalized_laplacian)
        laplacians.append(L)

    return laplacians, perms[0] if len(perms) > 0 else None


t_start = time.time()

np.random.seed(0)
n_rows_cols = 28
A, nodes_coordinates = grid_graph(n_rows_cols)
L, perm = coarsen(A, coarsening_levels)

print('Execution time: {:.2f}s'.format(time.time() - t_start))

graph.plot_spectrum(L)  # plateau is due to all the disconnected fake nodes
# !!!! the plateau is in correspondence of 1 and not 0 because we are using a variation of the normalized laplacian which admits a degree different from 0 even in
# correspondence of disconnected nodes (normalized laplacian D^-0.5 * L * D^-0.5 = I - D^-0.5 * A * D^-0.5 doesn't exist for D_i==0!).
# With normalized_laplacian=False the plateau is indeed in correspondence of 0 (as it should) !!!!
# %%
# Normalize Laplacian
L_norm = []
for k in range(len(L)):
    L_norm.append(L[k] - sp.eye(L[k].shape[0]))
graph.plot_spectrum(L_norm, ymin=-1)
# %%
# plot the constructed graph

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def newline(p1, p2):
    # draw a line between p1 and p2
    ax = plt.gca()
    l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color='r', linewidth=0.1)
    ax.add_line(l)
    return l


plt.figure(dpi=200)
plt.imshow(A.todense())
plt.title('Original adjacency matrix')

plt.figure(dpi=200)
plt.scatter(nodes_coordinates[:, 0] * n_rows_cols, nodes_coordinates[:, 1] * n_rows_cols)

A_row, A_col = A.nonzero()

for idx_e in range(len(A_row)):
    newline(nodes_coordinates[A_row[idx_e]] * n_rows_cols, nodes_coordinates[A_col[idx_e]] * n_rows_cols)
# %%
# loading of MNIST dataset

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(dir_data, one_hot=False)

train_data = mnist.train.images.astype(np.float32)
val_data = mnist.validation.images.astype(np.float32)  # the first 5K samples of the training dataset
# are used for validation
test_data = mnist.test.images.astype(np.float32)
train_labels = mnist.train.labels
val_labels = mnist.validation.labels
test_labels = mnist.test.labels

t_start = time.time()
train_data = coarsening.perm_data(train_data, perm)
val_data = coarsening.perm_data(val_data, perm)
test_data = coarsening.perm_data(test_data, perm)
print('Execution time: {:.2f}s'.format(time.time() - t_start))
del perm


# %%
class ChebNet:
    """
    The neural network model.
    """

    # Helper functions used for constructing the model
    def _weight_variable(self, shape, regularization=True):
        """Initializer for the weights"""

        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:  # append the loss of the current variable to the regularization term
            self.regularizers.append(tf.nn.l2_loss(var))
        return var

    def _bias_variable(self, shape, regularization=True):
        """Initializer for the bias"""

        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        return var

    def frobenius_norm(self, tensor):
        """Computes the frobenius norm for a given tensor"""

        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        frobenius_norm = tf.sqrt(tensor_sum)
        return frobenius_norm

    def count_no_weights(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('#weights in the model: %d' % (total_parameters,))

    # Modules used by the graph convolutional network
    def chebyshevConv(self, x, L, Fout, K):
        """Applies chebyshev polynomials over the graph (i.e. it makes a spectral convolution)"""

        N, M, Fin = x.get_shape()  # N is the number of images
        # M the number of vertices in the images
        # Fin the number of features
        N, M, Fin = int(N), int(M), int(Fin)

        # Transform to Chebyshev basis. Here we apply the chebyshev polynomials. For simplicity instead of operating
        # over the eigenvalues of the laplacians we apply directly the chebysev polynomials to the laplacians.
        # This makes everything more efficient if the laplacians are actually represented as sparse matrices.
        x0 = x  # L^0 * x = I * x
        x = tf.expand_dims(x0, 0)  # shape = 1 x N x M x Fin

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # shape = 1 x M x Fin*N
            return tf.concat([x, x_], 0)  # shape = 1 x N x M x Fin

        if K > 1:
            x0 = tf.transpose(x0, [1, 2, 0])
            x0 = tf.reshape(x0, [M, Fin * N])
            x1 = tf.sparse_tensor_dense_matmul(L, x0)  # shape = M x Fin*N <- L^1 * x
            x1 = tf.reshape(x1, [M, Fin, N])  # shape = M x Fin x N <- L^1 * x
            x1 = tf.transpose(x1, [2, 0, 1])  # shape = N x M x Fin
            x0 = tf.reshape(x0, [M, Fin, N])
            x0 = tf.transpose(x0, [2, 0, 1])  # shape = N x M x Fin
            x = concat(x, x1)  # shape = 2 x N x M x Fin
        for k in range(2, K):
            x1 = tf.transpose(x1, [1, 2, 0])
            x1 = tf.reshape(x1, [M, Fin * N])
            x2 = tf.sparse_tensor_dense_matmul(L, x1)  # shape = M x Fin*N <- L^k * x
            x2 = tf.reshape(x2, [M, Fin, N])  # shape = M x Fin x N <- L^1 * x
            x2 = tf.transpose(x2, [2, 0, 1])
            x1 = tf.reshape(x1, [M, Fin, N])
            x1 = tf.transpose(x1, [2, 0, 1])  # shape = N x M x Fin
            x2 = 2 * x2 - x0  # shape = N x M x Fin <- recursive definition of chebyshev polynomials
            x = concat(x, x2)  # shape = K x N x M x Fin
            x0, x1 = x1, x2
        x = tf.transpose(x, [1, 2, 3, 0])  # shape = N x M x Fin x K
        x = tf.reshape(x, [N * M, Fin * K])  # shape = N*M x Fin*K

        # Filter: Fout filters of order K applied over all the Fin features
        W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Applies bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)  # add the bias to the convolutive layer

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2 (this is possible thanks to the reordering we previously did)."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # shape = N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            return tf.squeeze(x, [3])  # shape = N x M/p x F
        else:
            return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    # function used for extracting the result of our model
    def _inference(self, x, dropout):  # definition of the model

        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(len(self.p)):
            with tf.variable_scope('cgconv{}'.format(i + 1)):
                with tf.name_scope('filter'):
                    x = self.chebyshevConv(x, self.L[i * 2], self.F[i], self.K[i])
                with tf.name_scope('bias_relu'):
                    x = self.b1relu(x)
                with tf.name_scope('pooling'):
                    x = self.mpool1(x, self.p[i])

        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M * F)])  # N x M
        for i, M in enumerate(self.M[:-1]):  # apply a fully connected layer for each layer defined in M
            # (we discard the last value in M since it contains the number of classes we have
            # to predict)
            with tf.variable_scope('fc{}'.format(i + 1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = self.fc(x, self.M[-1], relu=False)
        return x

    def convert_coo_to_sparse_tensor(self, L):
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data.astype('float32'), L.shape)
        L = tf.sparse_reorder(L)
        return L

    def __init__(self, p, K, F, M, M_0, batch_size, L,
                 decay_steps, decay_rate, learning_rate=1e-4, momentum=0.9, regularization=5e-4,
                 idx_gpu='/gpu:0'):
        self.regularizers = list()  # list of regularization l2 loss for multiple variables

        self.p = p  # dimensions of the pooling layers
        self.K = K  # List of polynomial orders, i.e. filter sizes or number of hops
        self.F = F  # Number of features of convolutional layers

        self.M = M  # Number of neurons in fully connected layers

        self.M_0 = M_0  # number of elements in the first graph

        self.batch_size = batch_size

        # definition of some learning parameters
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.regularization = regularization

        with tf.Graph().as_default() as g:
            self.graph = g
            tf.set_random_seed(0)
            with tf.device(idx_gpu):
                # definition of placeholders
                self.L = [self.convert_coo_to_sparse_tensor(c_L.tocoo()) for c_L in L]
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0), 'data')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

                # Model construction
                self.logits = self._inference(self.ph_data, self.ph_dropout)

                # Definition of the loss function
                with tf.name_scope('loss'):
                    self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                        labels=self.ph_labels)
                    self.cross_entropy = tf.reduce_mean(self.cross_entropy)
                with tf.name_scope('regularization'):
                    self.regularization *= tf.add_n(self.regularizers)
                self.loss = self.cross_entropy + self.regularization

                # Solver Definition
                with tf.name_scope('training'):
                    # Learning rate.
                    global_step = tf.Variable(0, name='global_step', trainable=False)  # used for counting how many iterations we have done
                    if decay_rate != 1:  # applies an exponential decay of the lr wrt the number of iterations done
                        learning_rate = tf.train.exponential_decay(
                            learning_rate, global_step, decay_steps, decay_rate, staircase=True)
                    # Optimizer.
                    if momentum == 0:
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                    else:  # applies momentum for increasing the robustness of the gradient
                        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
                    grads = optimizer.compute_gradients(self.loss)
                    self.op_gradients = optimizer.apply_gradients(grads, global_step=global_step)

                    # Computation of the norm gradients (useful for debugging)
                self.var_grad = tf.gradients(self.loss, tf.trainable_variables())
                self.norm_grad = self.frobenius_norm(tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0))

                # Extraction of the predictions and computation of accuracy
                self.predictions = tf.cast(tf.argmax(self.logits, dimension=1), tf.int32)
                self.accuracy = 100 * tf.contrib.metrics.accuracy(self.predictions, self.ph_labels)

                # Create a session for running Ops on the Graph.
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.session = tf.Session(config=config)

                # Run the Op to initialize the variables.
                init = tf.global_variables_initializer()
                self.session.run(init)

                self.count_no_weights()


# %%
# Convolutional parameters
p = [4, 4]  # Dimensions of the pooling layers
K = [25, 25]  # List of polynomial orders, i.e. filter sizes or number of hops
F = [32, 64]  # Number of features of convolutional layers

# FC parameters
C = max(mnist.train.labels) + 1  # Number of classes we have
M = [512, C]  # Number of neurons in fully connected layers

# Solver parameters
batch_size = 100
decay_steps = mnist.train.num_examples / batch_size  # number of steps to do before decreasing the learning rate
decay_rate = 0.95  # how much decreasing the learning rate
learning_rate = 0.02
momentum = 0.9
regularization = 5e-4

# Definition of keep probabilities for dropout layers
dropout_training = 0.5
dropout_val_test = 1.0
# %%
# Construction of the learning obj
M_0 = L[0].shape[0]  # number of elements in the first graph
learning_obj = ChebNet(p, K, F, M, M_0, batch_size, L_norm,
                       decay_steps, decay_rate,
                       learning_rate=learning_rate, regularization=regularization,
                       momentum=momentum)

# definition of overall number of training iterations and validation frequency
num_iter_val = 600
num_total_iter_training = 21000

num_iter = 0

list_training_loss = list()
list_training_norm_grad = list()
list_val_accuracy = list()
# %%
# training and validation
indices = collections.deque()  # queue that will contain a permutation of the training indexes
for k in range(num_iter, num_total_iter_training):

    # Construction of the training batch
    if len(indices) < batch_size:  # Be sure to have used all the samples before using one a second time.
        indices.extend(np.random.permutation(train_data.shape[0]))  # reinitialize the queue of indices
    idx = [indices.popleft() for i in range(batch_size)]  # extract the current batch of samples

    # data extraction
    batch_data, batch_labels = train_data[idx, :], train_labels[idx]

    feed_dict = {learning_obj.ph_data: batch_data,
                 learning_obj.ph_labels: batch_labels,
                 learning_obj.ph_dropout: dropout_training}

    # Training
    tic = time.time()
    _, current_training_loss, norm_grad = learning_obj.session.run([learning_obj.op_gradients,
                                                                    learning_obj.loss,
                                                                    learning_obj.norm_grad], feed_dict=feed_dict)
    training_time = time.time() - tic

    list_training_loss.append(current_training_loss)
    list_training_norm_grad.append(norm_grad)

    if (np.mod(num_iter, num_iter_val) == 0):  # validation
        msg = "[TRN] iter = %03i, cost = %3.2e, |grad| = %.2e (%3.2es)" \
              % (num_iter, list_training_loss[-1], list_training_norm_grad[-1], training_time)
        print(msg)

        # Validation Code
        tic = time.time()
        val_accuracy = 0
        for begin in range(0, val_data.shape[0], batch_size):
            end = begin + batch_size
            end = min([end, val_data.shape[0]])

            # data extraction
            batch_data = np.zeros((end - begin, val_data.shape[1]))
            batch_data = val_data[begin:end, :]
            batch_labels = np.zeros(batch_size)
            batch_labels[:end - begin] = val_labels[begin:end]

            feed_dict = {learning_obj.ph_data: batch_data,
                         learning_obj.ph_labels: batch_labels,
                         learning_obj.ph_dropout: dropout_val_test}

            batch_accuracy = learning_obj.session.run(learning_obj.accuracy, feed_dict)
            val_accuracy += batch_accuracy * batch_data.shape[0]
        val_accuracy = val_accuracy / val_data.shape[0]
        val_time = time.time() - tic
        msg = "[VAL] iter = %03i, acc = %4.2f (%3.2es)" % (num_iter, val_accuracy, val_time)
        print(msg)
    num_iter += 1
# %%
# Test code
tic = time.time()
test_accuracy = 0
for begin in range(0, test_data.shape[0], batch_size):
    end = begin + batch_size
    end = min([end, test_data.shape[0]])

    batch_data = np.zeros((end - begin, test_data.shape[1]))
    batch_data = test_data[begin:end, :]

    feed_dict = {learning_obj.ph_data: batch_data, learning_obj.ph_dropout: 1}

    batch_labels = np.zeros(batch_size)
    batch_labels[:end - begin] = test_labels[begin:end]
    feed_dict[learning_obj.ph_labels] = batch_labels

    batch_accuracy = learning_obj.session.run(learning_obj.accuracy, feed_dict)
    test_accuracy += batch_accuracy * batch_data.shape[0]
test_accuracy = test_accuracy / test_data.shape[0]
test_time = time.time() - tic
msg = "[TST] iter = %03i, acc = %4.2f (%3.2es)" % (num_iter, test_accuracy, test_time)
print(msg)
# %%

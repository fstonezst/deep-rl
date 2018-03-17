import tensorflow as tf
import tflearn
import numpy as np


class autoencoder:
    def __init__(self, sess, input_dim, h_dim=200, lr=1.0E-4, a_dim=1, lambda1=1):
        self.sess = sess
        self.s_dim, self.a_dim, self.h_dim = input_dim, a_dim, h_dim
        self.lambda1 = lambda1
        self.learning_rate = lr
        self.N_HIDDEN_1 = h_dim

        # encode
        self.inputs, self.feature_vector = self.create_decoder()

        # encode vector merge action
        self.action = tf.placeholder(tf.float32, [None, 1])
        t1, t2 = tflearn.activations.linear(self.feature_vector), tflearn.activations.linear(self.action)
        self.hideVector = tflearn.layers.merge_ops.merge([t1, t2], mode='concat')

        # decode
        self.stateOut, self.rewardOut = self.create_decoder(self.hideVector)

        # complete loss
        self.nextState = tf.placeholder(tf.float32, [None, self.s_dim])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.nextFeature = self.encode(self.nextState)
        self.loss = tflearn.mean_square(self.nextFeature, self.stateOut)
        self.loss += self.lambda1 * tflearn.mean_square(self.reward, self.rewardOut)

        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

    def train(self, inputs, nextState, reward):
        return self.sess.run([self.stateOut, self.rewardOut, self.optimize], feed_dict={
            self.inputs: inputs,
            self.reward: reward,
            self.nextState: nextState
        })

    def encode(self, inputs):
        return self.sess.run(self.feature_vector, feed_dict={
            self.inputs: inputs
        })

    def create_encoder(self, net_name='encoder'):
        N_HIDDEN_1 = self.N_HIDDEN_1

        # state input
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        inputLayer = tflearn.layers.normalization.batch_normalization(inputs, name='input_' + net_name + '_bn')

        # encoder layer
        w_init = tflearn.initializations.uniform(minval=-1 / np.sqrt(self.s_dim), maxval=1 / np.sqrt(self.s_dim))
        net = tflearn.fully_connected(inputLayer, N_HIDDEN_1, regularizer='L2', weight_decay=1.0E-2,
                                      weights_init=w_init, name='first_' + net_name + '_layer')
        net = tflearn.activation(net, 'relu')

        return inputs, net

    def create_decoder(self, net_name='deconder', inputs=None):
        h_dim = self.h_dim + self.a_dim

        # feature vector input
        # inputs = tflearn.input_data(shape=[None, self.h_dim])
        inputLayer = tflearn.layers.normalization.batch_normalization(inputs, name='input_' + net_name + '_bn')

        # decoder layer
        w_init = tflearn.initializations.uniform(minval=-1 / np.sqrt(h_dim), maxval=1 / np.sqrt(h_dim))
        state = tflearn.fully_connected(inputLayer, h_dim, regularizer='L2', weight_decay=1.0E-2,
                                        weights_init=w_init, name='first_' + net_name + '_state_layer')
        state = tflearn.activation(state, 'relu')

        w_init = tflearn.initializations.uniform(minval=-1 / np.sqrt(h_dim), maxval=1 / np.sqrt(h_dim))
        reward = tflearn.fully_connected(inputLayer, 1, regularizer='L2', weight_decay=1.0E-2,
                                         weights_init=w_init, name='first_' + net_name + '_reward_layer')
        reward = tflearn.activation(reward, 'sigmoid')

        return state, -reward

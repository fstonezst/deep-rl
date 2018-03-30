import tensorflow as tf
import tflearn
import numpy as np


class autoencoder:
    def __init__(self, sess, input_dim, h_dim=1, lr=1.0E-3, a_dim=1, lambda1=10):
        self.sess = sess
        self.s_dim, self.a_dim, self.h_dim = input_dim, a_dim, h_dim
        self.lambda1 = lambda1
        self.learning_rate = lr
        self.state_dim = 4

        # single layer encode
        self.inputs, self.feature_vector = self.create_encoder()

        # single layer decode
        self.action, self.stateOut, self.rewardOut = self.create_decoder(inputLayer=self.feature_vector)

        # complete state loss
        # self.nextState = tf.placeholder(tf.float32, [None, self.h_dim], name='next_state_input')
        # self.nextState = tf.placeholder(tf.float32, [None, self.s_dim], name='next_state_input')
        self.nextState = tf.placeholder(tf.float32, [None, self.state_dim], name='next_state_input')
        self.state_loss = tflearn.mean_square(self.nextState, self.stateOut)

        # complete reward loss
        self.reward = tf.placeholder(tf.float32, [None, 1], name='reward_input')
        self.reward_loss = self.lambda1 * tflearn.mean_square(self.reward, self.rewardOut)

        # sum loss
        # self.loss = self.reward_loss + self.state_loss
        # self.loss = self.reward_loss
        self.loss = self.state_loss

        # optimizer
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

    def train(self, inputs, action, nextState, reward):
        return self.sess.run([self.stateOut, self.rewardOut, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.reward: reward,
            self.nextState: nextState
        })

    def encode(self, nextState):
        return self.sess.run(self.feature_vector, feed_dict={
            self.inputs: nextState
        })

    def create_encoder(self, net_name='encoder'):
        N_HIDDEN_1, N_HIDDEN_2 = self.h_dim, self.s_dim
        # N_HIDDEN_1, N_HIDDEN_2 = self.h_dim, self.h_dim


        # state input
        inputs = tflearn.input_data(shape=[None, self.s_dim], name='state_input')
        inputLayer = tflearn.layers.normalization.batch_normalization(inputs, name='input_' + net_name + '_bn')

        # encoder layer
        w_init = tflearn.initializations.uniform(minval=-1 / np.sqrt(self.s_dim), maxval=1 / np.sqrt(self.s_dim))
        net = tflearn.fully_connected(inputs, N_HIDDEN_1,
                                      regularizer='L2', weight_decay=1.0E-2,
                                      weights_init=w_init, name='first_' + net_name + '_layer',activation='relu')
        net = tflearn.layers.normalization.batch_normalization(net, name='first_' + net_name + '_bn')
        net = tflearn.activation(net, 'relu')

        # w_init = tflearn.initializations.uniform(minval=-1 / np.sqrt(N_HIDDEN_1), maxval=1 / np.sqrt(N_HIDDEN_1))
        # net = tflearn.fully_connected(net, N_HIDDEN_2,
        #                               regularizer='L2', weight_decay=1.0E-2,
        #                               weights_init=w_init, name='second_' + net_name + '_layer')
        # net = tflearn.layers.normalization.batch_normalization(net, name='second_' + net_name + '_bn')
        # net = tflearn.activation(net, 'relu')


        return inputs, net

    def create_decoder(self, net_name='deconder', inputLayer = None):
        N_HIDDEN_1, N_HIDDEN_2 = 256, self.h_dim + self.a_dim
        # N_HIDDEN_1, N_HIDDEN_2 = self.h_dim, self.h_dim + self.a_dim
        input_dim = self.h_dim + self.a_dim
        state_dim = self.state_dim

        # action input
        action = tflearn.input_data(shape=[None, self.a_dim], name='action_input')

        # merge action input and feature vector as a input vector of decoder layer
        t1, t2 = tflearn.activations.linear(inputLayer), tflearn.activations.linear(action)
        hideVector = tflearn.layers.merge_ops.merge([t1, t2], mode='concat')

        # decoder layer
        # w_init = tflearn.initializations.uniform(minval=-1 / np.sqrt(input_dim), maxval=1 / np.sqrt(input_dim))
        # hideVector = tflearn.fully_connected(hideVector, N_HIDDEN_1,
        #                                 regularizer='L2', weight_decay=1.0E-2,
        #                                 weights_init=w_init, name='first_' + net_name + '_state_layer')
        # hideVector = tflearn.layers.normalization.batch_normalization(hideVector, name='first_' + net_name + '_bn')
        # hideVector = tflearn.activation(hideVector, 'relu')

        # w_init = tflearn.initializations.uniform(minval=-1 / np.sqrt(N_HIDDEN_1), maxval=1 / np.sqrt(N_HIDDEN_1))
        w_init = tflearn.initializations.uniform(minval=-1 / np.sqrt(input_dim), maxval=1 / np.sqrt(input_dim))
        # state = tflearn.fully_connected(hideVector, self.h_dim,
        # state = tflearn.fully_connected(hideVector, self.s_dim,
        state = tflearn.fully_connected(hideVector, state_dim,
                                        regularizer='L2', weight_decay=1.0E-2,
                                        weights_init=w_init, name='out_' + net_name + '_state_layer')
        # state = tflearn.activation(state, 'relu')
        # state = tflearn.activation(state, 'sigmoid')

        # w_init = tflearn.initializations.uniform(minval=-1 / np.sqrt(N_HIDDEN_1), maxval=1 / np.sqrt(N_HIDDEN_1))
        w_init = tflearn.initializations.uniform(minval=-1 / np.sqrt(input_dim), maxval=1 / np.sqrt(input_dim))
        reward = tflearn.fully_connected(hideVector, 1, regularizer='L2',
                                         weight_decay=1.0E-2,
                                         weights_init=w_init, name='out_' + net_name + '_reward_layer')
        reward = tflearn.activation(reward, 'sigmoid')

        return action, state, -reward

import tensorflow as tf
import tflearn
import math


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        times = 3
        N_HIDDEN_1, N_HIDDEN_2 = 400 * times, 300 * times
        DROPOU_KEEP_PROB = 0.5
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        # net = tflearn.fully_connected(inputs, N_HIDDEN_1, activation='relu',regularizer='L2')
        net = tflearn.fully_connected(inputs, N_HIDDEN_1, activation=self.swish, regularizer='L2')
        net = tflearn.dropout(net, DROPOU_KEEP_PROB)
        net = tflearn.layers.normalization.batch_normalization(net)

        # net = tflearn.fully_connected(net, N_HIDDEN_2, activation='relu',regularizer='L2')
        net = tflearn.fully_connected(net, N_HIDDEN_2, activation=self.swish,regularizer='L2')
        net = tflearn.dropout(net, DROPOU_KEEP_PROB)
        net = tflearn.layers.normalization.batch_normalization(net)

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-3.0E-3, maxval=3.0E-3)
        b_init = tflearn.initializations.uniform(minval=-3.0E-3, maxval=3.0E-3)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init, bias_init=b_init)

        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def swish(self,x):
        return x * tflearn.activations.sigmoid(x)

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

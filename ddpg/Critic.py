import tensorflow as tf
import tflearn

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # with tf.name_scope('loss'):
        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)

        # self.loss = -tf.reduce_mean(tf.clip_by_value((-1 * self.predicted_q_value), 10E-10, 1) * tf.log(tf.clip_by_value((-1.0 * self.out), 10E-10, 1)))
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)
        # self.loss_summary = tf.summary.scalar('loss', self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        times = 3
        N_HIDDEN_1, N_HIDDEN_2  = 400 * times, 300 * times
        DROPOU_KEEP_PROB= 0.5

        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, N_HIDDEN_1, activation='relu',regularizer='L2')
        net = tflearn.dropout(net, DROPOU_KEEP_PROB)
        net = tflearn.layers.normalization.batch_normalization(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, N_HIDDEN_2, regularizer='L2', activation='relu')
        t1 = tflearn.dropout(t1, DROPOU_KEEP_PROB)
        t1 = tflearn.layers.normalization.batch_normalization(t1)

        t2 = tflearn.fully_connected(action, N_HIDDEN_2, regularizer='L2', activation='relu')
        t2 = tflearn.dropout(t2, DROPOU_KEEP_PROB)
        t2 = tflearn.layers.normalization.batch_normalization(t2)

        # net = tflearn.activation(
        #     tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        t1, t2 = tflearn.activations.linear(t1), tflearn.activations.linear(t2)
        net = tflearn.layers.merge_ops.merge([t1, t2], mode='elemwise_sum')
        net = tflearn.fully_connected(net, N_HIDDEN_2, regularizer='L2', activation='relu')
        net = tflearn.dropout(net, DROPOU_KEEP_PROB)
        net = tflearn.layers.normalization.batch_normalization(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        # return self.sess.run([self.loss_summary, self.out, self.optimize], feed_dict={
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def getOut(self, inputs, action):
        return self.sess.run([self.out], feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    # def getLoss(self, out, predicted_q_value):
    #     return self.sess.run([self.loss], feed_dict={
    #         self.out:out
    #
    #     })




""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp
from pathfollowing_env import PathFollowing
from pathfollowing_env_v2 import PathFollowingV2
from replay_buffer import ReplayBuffer
import os
import random


# ===========================
#   Actor and Critic DNNs
# ===========================

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
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400 * times, activation='relu',regularizer='L2')
        # net = tflearn.fully_connected(inputs, 400, activation='relu')
        # net = tflearn.fully_connected(inputs, 800, activation='relu')
        # net = tflearn.fully_connected(inputs, 50, activation='relu')
        net = tflearn.dropout(net, 0.5)
        # net = tflearn.fully_connected(inputs, 800, activation='relu')

        net = tflearn.layers.normalization.batch_normalization(net)

        # net = tflearn.fully_connected(net, 300, activation='relu')
        net = tflearn.fully_connected(inputs, 300 * times, activation='relu',regularizer='L2')
        # net = tflearn.fully_connected(net, 600, activation='relu')
        # net = tflearn.fully_connected(net, 30, activation='relu')
        net = tflearn.dropout(net, 0.5)

        net = tflearn.layers.normalization.batch_normalization(net)

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)

        # out = tflearn.fully_connected(
        #     net, self.a_dim, activation='tanh', weights_init=w_init)

        out1 = tflearn.fully_connected(
            net, 1, activation='tanh', weights_init=w_init)
        out2 = tflearn.fully_connected(
            net, 1, activation='sigmoid', weights_init=w_init)
        out = tflearn.merge_outputs([out1,out2])
        # out = np.array([out1,out2])
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

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
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        # net = tflearn.fully_connected(inputs, 400)
        # net = tflearn.fully_connected(inputs, 800)
        net = tflearn.fully_connected(inputs, 400 * times, activation='relu',regularizer='L2')
        # net = tflearn.fully_connected(inputs, 50)
        net = tflearn.dropout(net, 0.5)
        # net = tflearn.fully_connected(inputs, 800)
        net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300 * times, regularizer='L2')
        # t1 = tflearn.fully_connected(net, 30)
        # t1 = tflearn.dropout(t1,0.5)
        t2 = tflearn.fully_connected(action, 300 * times, regularizer='L2')
        # t2 = tflearn.fully_connected(action, 30)
        # t2 = tflearn.dropout(t2,0.5)
        # t1 = tflearn.fully_connected(net, 600)
        # t2 = tflearn.fully_connected(action, 600)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

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

    # def getLoss(self, out, predicted_q_value):
    #     return self.sess.run([self.loss], feed_dict={
    #         self.out:out
    #
    #     })

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


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    train_loss = tf.Variable(0.)
    tf.summary.scalar("train_loss", train_loss)
    error = tf.Variable(0.)
    tf.summary.scalar("error", error)

    summary_vars = [episode_reward, episode_ave_max_q, train_loss, error]
    summary_ops = tf.summary.merge_all()
    # summary_ops = tf.summary.merge_all_summaries()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    totalTime = 0
    ave_error = 0
    for i in range(int(args['max_episodes'])):
        if totalTime > int(args['max_episodes_len']):
            break

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        ave_diff = 0
        total_loss = 0
        total_noise = np.array([0.0,0.0])

        # for j in range(int(args['max_episode_len'])):

        for j in range(4000):
            if args['render_env']:
                env.render()

            # Added exploration noise
            # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))

            # noise = actor_noise() / 2
            # if i < 40:
            #     noise = actor_noise() / 10
            # else:
            #     noise = actor_noise()

            # if i > 40:
            #     noise = noise / 10

            # a = actor.predict(np.reshape(s, (1, actor.s_dim))) + noise
            dirOut = actor.predict(np.reshape(s, (1, actor.s_dim)))
            # while noise / dirOut[0] > 0.2:
            #     noise /= 10
            noise = actor_noise()
            # if abs(ave_error) < 0.0001:
            if abs(ave_diff) < 0.0001:
                noise = np.zeros(noise.shape,noise.dtype)
            else:
                # while abs(noise) > abs(ave_error * 0.5):
                while abs(sum(noise)) > abs(float(ave_diff)):
                    noise *= 0.8
            total_noise += noise
            a = dirOut + noise

            # print "noise:"+ str(noise), "a:"+str(a)

            s2, r, terminal, info = env.step(a)

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                y_label = np.reshape(y_i, (int(args['minibatch_size']), 1))

                # Update the critic given the targets
                # merged = tf.merge_all_summaries()

                # loss = sess.run([critic.loss], feed_dict={
                #     critic.predicted_q_value: np.reshape(y_i, (int(args['minibatch_size']), 1)),
                #     critic.inputs: s_batch,
                #     critic.action: a_batch
                # })
                #
                # predicted_q_value = sess.run([critic.optimize], feed_dict={
                #     critic.loss: loss
                # })

                predicted_q_value, _ = critic.train(
                    # s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))
                    s_batch, a_batch, y_label)

                loss = sess.run([critic.loss], feed_dict={
                    critic.inputs: s_batch,
                    critic.action: a_batch,
                    critic.predicted_q_value: y_label
                })


                diff = y_label - predicted_q_value
                diff = abs(diff)
                ave_diff = sum(diff)[0] / float(len(diff))

                ep_ave_max_q += np.amax(predicted_q_value)
                total_loss += np.amax(loss)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                # import matplotlib.pyplot as plt
                # print("ave_diff:"+str(ave_diff))
                summary_str = sess.run(summary_ops, feed_dict={
                    # summary_vars[0]: ep_reward ,
                    summary_vars[0]: ep_reward / float(j),
                    summary_vars[1]: ep_ave_max_q / float(j),
                    summary_vars[2]: total_loss / float(j),
                    summary_vars[3]: info.get("avgError")[0]
                })

                writer.add_summary(summary_str, i)

                writer.flush()

                totalTime += j  # info.get("times")

                # if j > 750 and i % 5 == 0:
                # if totalTime > 10000:
                #     list = info.get("result")
                #     fig = plt.figure()
                #     ax1 = fig.add_subplot(1, 1, 1)
                #     ax1.plot(list[0], 'g-', label='Route')
                #     ax1.plot(list[1], 'o-',color='red', label='Move')
                #     plt.show()

                if j > 0:
                    ave_error = info.get("avgError")[0]
                    print (total_noise / float(j))
                    # print (total_loss/float(j))
                    print(
                    '| Reward: {:.4f} | Episode: {:d} | times:{:d} | Qmax: {:.4f} | ave_error: {:.4f} | ave_diff: {:.4f} | loss: {:.4f}'.format(
                        int(ep_reward) / float(j), i, totalTime, (ep_ave_max_q / float(j)), ave_error, float(ave_diff), (total_loss / float(j))))
                break
    writer.close()


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(random.randint(0, 1))
    with tf.Session() as sess:

        # env = gym.make(args['env'])
        # env = PathFollowing()
        env = PathFollowingV2()
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = len(env.reset()) #.shape[1] #8  # env.observation_space.shape[0]
        # state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, actor_noise)

        if args['use_gym_monitor']:
            env.close()
            # env.monitor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    # parser.set_defaults(render_env=True)

    parser.set_defaults(use_gym_monitor=True)
    parser.set_defaults(max_episodes=120)
    parser.set_defaults(max_episodes_len=80000)
    # parser.set_defaults(minibatch_size=64)
    parser.set_defaults(minibatch_size=128)
    # parser.set_defaults(env='PathFollowing-v0')
    # parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)

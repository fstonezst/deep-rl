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
import sys
sys.path.append("/home/CAD409/my_code/lib/python2.7/site-packages/gym-0.9.3-py2.7.egg")
from gym import wrappers
import argparse
import pprint as pp
from pathfollowing_env_v2 import PathFollowingV2
from pathfollowing_env_v1 import PathFollowingV1
from pathfollowing_env_v3 import PathFollowingV3
from replay_buffer import ReplayBuffer
from autoencoder import autoencoder
import os
from AGV_Model import AGV


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    ae_reward_loss = tf.Variable(0.)
    tf.summary.scalar("ae_reward_loss", ae_reward_loss)
    ae_total_loss = tf.Variable(0.)
    tf.summary.scalar("ae_total_loss", ae_total_loss)

    summary_vars = [ae_reward_loss, ae_total_loss]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================
def trainAE(sess, env, args, s_dim, a_dim, ae):
    from Noise import OrnsteinUhlenbeckNoise
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)
    saver = tf.train.Saver()
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    totalTime = 0
    count = 10
    last_loss = 4.0E8
    isConvergence = True
    orientationNoise = 0
    oriNoiseRate = 1
    for i in range(int(args['max_episodes'])):
        if totalTime > int(args['max_episodes_len']):
            break

        s = env.reset()

        ae_r_loss_sum = 0
        ae_total_loss_sum = 0


        # DELTA  The rate of change (time)
        # SIGMA  Volatility of the stochastic processes
        # OU_A   The rate of mean reversion
        # OU_MU  The long run average interest rate
        orientationN = OrnsteinUhlenbeckNoise(delta=0.5, sigma=0.5 * AGV.MAX_ORIENTATION * oriNoiseRate)

        if last_loss > 1.0E-3:
           isConvergence = False
           count = 10
        else:
            count -= 1
            if count == 0:
                saver.save(sess,'ae_model_'+str(i))
                print "===================="+str(i)+"================="
                break

        for j in range(1, 4000):
            if args['render_env']:
                env.render()

            orientationNoise = orientationN.ornstein_uhlenbeck_level(orientationNoise)
            noise = np.array([[orientationNoise]])
            a = noise

            s2, r, terminal, info = env.step(a)

            replay_buffer.add(np.reshape(s, (s_dim,)), np.reshape(a, (a_dim,)), r,
                              terminal, np.reshape(s2, (s_dim,)))

            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                r_i = []

                for k in range(int(args['minibatch_size'])):
                    r_i.append(r_batch[k])

                r_label = np.reshape(r_i, (int(args['minibatch_size']), 1))

                nextState = ae.encode(s2_batch)
                ae.train(s_batch, a_batch, nextState, r_label)

                ae_reward_loss = sess.run([ae.reward_loss], feed_dict={
                    ae.inputs: s_batch,
                    ae.action: a_batch,
                    ae.reward: r_label,
                    ae.nextState: nextState
                })

                ae_total_loss = sess.run([ae.loss], feed_dict={
                    ae.inputs: s_batch,
                    ae.action: a_batch,
                    ae.reward: r_label,
                    ae.nextState: nextState
                })

                ae_r_loss_sum += np.amax(ae_reward_loss)
                ae_total_loss_sum += np.amax(ae_total_loss)

            s = s2

            if terminal:

                totalTime += j

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ae_r_loss_sum / float(j),
                    summary_vars[1]: ae_total_loss_sum / float(j)
                })
                last_loss = ae_total_loss_sum/float(j)

                writer.add_summary(summary_str, i)

                writer.flush()
                if i % 100 == 0:
                    print '===='+str(i)+':'+str(totalTime)+'===='

                break

    if not isConvergence:
        saver.save(sess, 'model_0')
    writer.close()

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(random.randint(0, 1))
    with tf.Session() as sess:

        if args['envno'] == '2':
            env = PathFollowingV2()
        elif args['envno'] == '1':
            env = PathFollowingV1()
        elif args['envno'] == '3':
            env = PathFollowingV3()
        else:
            env = PathFollowingV3()

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = len(env.reset()) #.shape[1] #8  # env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        ae = autoencoder(sess, input_dim=state_dim, h_dim=200, lr=1.0E-3,a_dim=action_dim, lambda1=10)
        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        if args['model'] == '':
            trainAE(sess, env, args, state_dim, action_dim, ae)

        if args['use_gym_monitor']:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=1.0E-4)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=1.0E-3)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1.0E4)
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
    parser.add_argument('--gpu', help='gpu index', default='0')
    parser.add_argument('--model', help='restore model', default='')
    parser.add_argument('--envno', help='env NO.', default='3')
    parser.add_argument('--debug', help='store path', default='False')

    parser.set_defaults(render_env=False)

    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(max_episodes=2.0E4)
    parser.set_defaults(max_episodes_len=1.0E6)
    parser.set_defaults(minibatch_size=128)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)

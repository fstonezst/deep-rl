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
    ae_state_loss = tf.Variable(0.)
    tf.summary.scalar("ae_state_loss", ae_state_loss)
    reward_ave = tf.Variable(0.)
    tf.summary.scalar("reward", reward_ave)

    summary_vars = [ae_reward_loss, ae_state_loss, reward_ave]
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
    last_state_loss, last_reward_loss = 4.0E8, 4.0E8
    isConvergence = True
    orientationNoise = 0
    oriNoiseRate = 1
    for i in range(int(args['max_episodes'])):
        if totalTime > int(args['max_episodes_len']):
            break

        s = env.reset()

        ae_r_loss_sum = 0
        ae_state_loss_sum = 0
        max_u1 = 0


        # DELTA  The rate of change (time)
        # SIGMA  Volatility of the stochastic processes
        # OU_A   The rate of mean reversion
        # OU_MU  The long run average interest rate
        orientationN = OrnsteinUhlenbeckNoise(delta=0.5, sigma=0.5 * AGV.MAX_ORIENTATION * oriNoiseRate)
        reward_min, reward_sum = 0, 0

        max_pr, min_pr = 0, 10

        isConvergence = True
        if last_state_loss > 1.0E-5 or last_reward_loss > 1.0E-4 or i < 100:
           isConvergence = False
           count = 3
        else:
            # count -= 1
            # if count == 0:
            saver.save(sess, 'model_ae_'+str(i))
            print "===================="+str(i)+"================="
            break

        for j in range(1, 4000):
            if args['render_env']:
                env.render()

            orientationNoise = orientationN.ornstein_uhlenbeck_level(orientationNoise)
            noise = np.array([[orientationNoise]])
            a = noise


            s2, r, terminal, info = env.step(a)
            if r < reward_min:
                reward_min = r
            reward_sum += r

            new_s2, state_dim = [], 4

            index_i = env.historyLength-1
            new_s2.append(s2[index_i] + 0.5)
            index_i+=env.historyLength
            new_s2.append((s2[index_i] - AGV.MIN_ANGLE)/(AGV.MAX_ANGLE - AGV.MIN_ANGLE))
            index_i+=env.historyLength
            new_s2.append(1)
            index_i+=env.historyLength
            new_s2.append((s2[index_i]+0.1)/0.2)
            if new_s2[-1] > max_u1:
                max_u1 = new_s2[-1]

            # for index in range(env.historyLength-1, s_dim+1, env.historyLength):
            #     new_s2.append(s2[index])
            new_s2 = np.array(new_s2)

            replay_buffer.add(np.reshape(s, (s_dim,)), np.reshape(a, (a_dim,)), r,
                              terminal, np.reshape(new_s2, (state_dim,)))
            # replay_buffer.add(np.reshape(s, (s_dim,)), np.reshape(a, (a_dim,)), r,
            #                   terminal, np.reshape(s2, (s_dim,)))

            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                r_i = []

                for k in range(int(args['minibatch_size'])):
                    r_i.append(r_batch[k])

                r_label = np.reshape(r_i, (int(args['minibatch_size']), 1))

                # nextState = ae.encode(s2_batch)
                nextState = s2_batch
                ae.train(s_batch, a_batch, nextState, r_label)

                ae_reward_loss = sess.run([ae.reward_loss], feed_dict={
                    ae.inputs: s_batch,
                    ae.action: a_batch,
                    ae.reward: r_label,
                    ae.nextState: nextState
                })

                ae_state_loss = sess.run([ae.state_loss], feed_dict={
                    ae.inputs: s_batch,
                    ae.action: a_batch,
                    ae.reward: r_label,
                    ae.nextState: nextState
                })

                pre_reward = ae.predict_reward(s_batch, a_batch)
                max, min = np.amax(pre_reward), np.amin(pre_reward)
                if max > max_pr:
                    max_pr = max
                if min < min_pr:
                    min_pr = min
                ae_r_loss_sum += np.amax(ae_reward_loss)
                ae_state_loss_sum += np.amax(ae_state_loss)

            s = s2

            if terminal:

                totalTime += j

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ae_r_loss_sum / float(j),
                    summary_vars[1]: ae_state_loss_sum / float(j),
                    summary_vars[2]: reward_sum / float(j)
                })
                last_state_loss = ae_state_loss_sum / float(j)
                last_reward_loss = ae_r_loss_sum / float(j)

                writer.add_summary(summary_str, i)

                writer.flush()
                if i % 100 == 0:
                    print '===='+str(i)+':'+str(totalTime)+':'+str(reward_min)+':'+str(min_pr)+'===='

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

        ae = autoencoder(sess, input_dim=state_dim, a_dim=action_dim)
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
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1.0E6)
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
    parser.set_defaults(minibatch_size=256)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)

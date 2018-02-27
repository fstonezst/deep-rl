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
from replay_buffer import ReplayBuffer
from Critic import CriticNetwork
from Actor import ActorNetwork
import os
import random
from AGV_Model import AGV
import csv


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
    train_g = tf.Variable(0.)
    tf.summary.scalar("gradient", train_g)
    train_g2 = tf.Variable(0.)
    tf.summary.scalar("gradient2", train_g2)

    summary_vars = [episode_reward, episode_ave_max_q, train_loss, error, train_g, train_g2]
    summary_ops = tf.summary.merge_all()
    # summary_ops = tf.summary.merge_all_summaries()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic):
    from Noise import OrnsteinUhlenbeckNoise
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    saver = tf.train.Saver()

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    totalTime = 0
    count = 10
    curr_model_no = 0
    oriNoiseRate, rotNoiseRate = 1, 0.8
    last_loss, last_times, lastReward, last_error = 4.0E8, 0, -1, 4
    isConvergence = True

    for i in range(int(args['max_episodes'])):
        if totalTime > int(args['max_episodes_len']):
            break

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        total_loss = 0
        total_gradient = 0
        total_gradient2 = 0


        # DELTA  The rate of change (time)
        # SIGMA  Volatility of the stochastic processes
        # OU_A   The rate of mean reversion
        # OU_MU  The long run average interest rate
        orientationN = OrnsteinUhlenbeckNoise(delta=0.5, sigma=0.5 * AGV.MAX_ORIENTATION * oriNoiseRate)
        rotationN = OrnsteinUhlenbeckNoise(delta=0.5, sigma=0.5 * AGV.MAX_ROTATION * rotNoiseRate)

        total_noise0, total_noise1 = [], []
        orientationNoise, rotationNoise = 0, 0

        isConvergence = True
        # if last_loss > 4.0E-3 or last_error > 0.05 or last_times < env.max_time or lastReward < -0.01 or i < 500:
        if last_loss > 4.0E-3 or last_times < env.max_time or lastReward < -0.005 or i < 500:
           isConvergence = False
           count = 10
           if lastReward > -0.16 and i > (curr_model_no + 30):
               curr_model_no = i
               saver.save(sess, 'model_'+str(i))
        else:
            count -= 1
            if count == 0:
                #     env.setCarMess(500 + random.randint(100, 500))
                #     env.car.Ir, env.car.w_mss, env.car.Ip1 = [18, 1, 1], [15, 1.8, 1.8], 17
                #     env.car.Ir, env.car.w_mss, env.car.Ip1 = [13, 0.03, 0.03], [20, 2.3, 2.3], 10
                #     print "===================="+str(env.car.mess)+"================="
                #     count = 10
                saver.save(sess,'model_'+str(i))
                break

        for j in range(1, 4000):
            if args['render_env']:
                env.render()

            # Added exploration noise
            dirOut = actor.predict(np.reshape(s, (1, actor.s_dim)))

            if not isConvergence:
                orientation,orientationNoise = dirOut[0][0], orientationN.ornstein_uhlenbeck_level(orientationNoise)
                rotation, rotationNoise = dirOut[0][1], rotationN.ornstein_uhlenbeck_level(rotationNoise)
                if abs(orientation) <= 0.01:
                    orientationNoise = 0
                while abs(orientationNoise) > abs(orientation) * oriNoiseRate:
                    orientationNoise /= 2
                if abs(rotation) <= 1:
                    rotationNoise = 0
                while abs(rotationNoise) > abs(rotation) * rotNoiseRate:
                    rotationNoise /= 2
                noise = np.array([orientationNoise, rotationNoise])
                total_noise0.append(orientationNoise)
                total_noise1.append(rotationNoise)
                a = dirOut + noise
            else:
                a = dirOut

            s2, r, terminal, info = env.step(a)

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))
            lastReward = r
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
                if not isConvergence:
                    predicted_q_value, _ = critic.train(
                        s_batch, a_batch, y_label)
                else:
                    predicted_q_value = critic.predict(s_batch, a_batch)

                loss = sess.run([critic.loss], feed_dict={
                    critic.inputs: s_batch,
                    critic.action: a_batch,
                    critic.predicted_q_value: y_label
                })

                ep_ave_max_q += np.amax(predicted_q_value)
                total_loss += np.amax(loss)

                # Update the actor policy using the sampled gradient
                aGrads1, aGrads2 = None,None
                if not isConvergence:
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])

                    aGrads1 = map(lambda x:x[0],grads[0])
                    aGrads2 = map(lambda x:x[1],grads[0])

                if np.amax(aGrads1) is not None:
                    total_gradient += np.amax(aGrads1)
                if np.amax(aGrads2) is not None:
                    total_gradient2 += np.amax(aGrads2)

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                totalTime += j

                action_r, action_s = info.get("action")[0], info.get("action")[1]
                speed = info.get("speed")
                avgError = info.get("avgError")

                summary_str = sess.run(summary_ops, feed_dict={
                    # summary_vars[0]: ep_reward ,
                    summary_vars[0]: ep_reward / float(j),
                    summary_vars[1]: ep_ave_max_q / float(j),
                    summary_vars[2]: total_loss / float(j),
                    summary_vars[3]: avgError,
                    summary_vars[4]: total_gradient / float(j),
                    summary_vars[5]: total_gradient2 / float(j)
                })

                writer.add_summary(summary_str, i)

                writer.flush()

                debug = True if str(args['debug']) == "True" else False
                if debug:
                    moveStorex, moveStorey= info.get("moveStore")[0], info.get("moveStore")[1]
                    wheelx, wheely = info.get("wheel")[0], info.get("wheel")[1]
                    error_record = info.get("error")
                    # if (not isConvergence and i > 450 and (i % 30 == 0)) or (isConvergence and (i % 20 == 0)):
                    if (not isConvergence and (i % 100 == 0)) or (isConvergence and (i % 10 == 0)):

                        with open('movePath'+str(i)+'.csv','wb') as f:
                            csv_writer = csv.writer(f)
                            for x,y in zip(moveStorex,moveStorey):
                                csv_writer.writerow([x,y])

                        with open('wheelPath'+str(i)+'.csv','wb') as f:
                            csv_writer = csv.writer(f)
                            for x,y in zip(wheelx,wheely):
                                csv_writer.writerow([x,y])

                        with open('action'+str(i)+'.csv','wb') as f:
                            csv_writer = csv.writer(f)
                            for x,y in zip(action_r,action_s):
                                csv_writer.writerow([x,y])

                        with open('error'+str(i)+'.csv','wb') as f:
                            csv_writer = csv.writer(f)
                            for x in error_record:
                                csv_writer.writerow([x])

                if j > 0 and i % 100 == 0:
                    if not isConvergence:
                        print max(total_noise0), min(total_noise0), (sum(total_noise0) / float(j))
                        print max(total_noise1), min(total_noise1), (sum(total_noise1) / float(j))
                    print(
                        'Reward: {:.4f} | Episode: {:d} | times:{:d} | max_r: {:.4f} | min_r: {:.4f}| max_s: {:.4f}| min_s: {:.4f}| ave_error: {:.4f} | ave_speed: {:.4f} | max_speed: {:.4f}'.format(
                           int(ep_reward) / float(j), i, totalTime, max(action_r), min(action_r), max(action_s), min(action_s), avgError, sum(speed)/float(j), max(speed)))
                last_loss = total_loss / float(j)
                last_error = avgError
                last_times = j
                break
    if not isConvergence:
        saver.save(sess, 'model_final')
    writer.close()

def predictWork(sess, model, env, args, actor):

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model)
    no = model.split('_')[1]
    times = 600

    for i in range(1):
        s, info, len = env.reset(), None, 0
        env.setMaxTime(times-10)
        for j in range(1, times):
            if args['render_env']:
                env.render()

            a = actor.predict(np.reshape(s, (1, actor.s_dim)))
            s, r, terminal, info = env.step(a)

            if terminal:
                len = j
                break

        moveStorex, moveStorey= info.get("moveStore")[0], info.get("moveStore")[1]
        wheelx, wheely = info.get("wheel")[0], info.get("wheel")[1]
        action_r, action_s = info.get("action")[0], info.get("action")[1]
        speed = info.get("speed")
        error_record = info.get("error")

        # no = str(i)

        with open('movePath'+no+'.csv','wb') as f:
            csv_writer = csv.writer(f)
            for x,y in zip(moveStorex,moveStorey):
                csv_writer.writerow([x,y])

        with open('wheelPath'+no+'.csv','wb') as f:
            csv_writer = csv.writer(f)
            for x,y in zip(wheelx,wheely):
                csv_writer.writerow([x,y])

        with open('action'+no+'.csv','wb') as f:
            csv_writer = csv.writer(f)
            for x,y in zip(action_r,action_s):
                csv_writer.writerow([x,y])

        with open('error'+no+'.csv','wb') as f:
            csv_writer = csv.writer(f)
            for x in error_record:
                csv_writer.writerow([x])

        if len > 0 and i % 1000 == 0:
            ave_error = info.get("avgError")
            print(
                # 'Reward: {:.4f} | Episode: {:d} | times:{:d} | max_r: {:.4f} | min_r: {:.4f}| max_s: {:.4f}| min_s: {:.4f}| ave_error: {:.4f} | ave_speed: {:.4f} | max_speed: {:.4f}'.format(
                'Episode: {:d} | times:{:d} | ave_error: {:.4f} | ave_speed: {:.4f}'.format(
                    i, len,  ave_error, sum(speed)/float(j)))

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(random.randint(0, 1))
    with tf.Session() as sess:

        # env = gym.make(args['env'])
        # env = PathFollowing()
        if args['envno'] == '2':
            env = PathFollowingV2()
        elif args['envno'] == '1':
            env = PathFollowingV1()
        else:
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

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        if args['model'] == '':
            train(sess, env, args, actor, critic)
        else:
            predictWork(sess, 'model_'+str(args['model']), env, args, actor)

        if args['use_gym_monitor']:
            env.close()
            # env.monitor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    # parser.add_argument('--actor-lr', help='actor network learning rate', default=1.0E-4)
    # parser.add_argument('--critic-lr', help='critic network learning rate', default=1.0E-3)
    parser.add_argument('--actor-lr', help='actor network learning rate', default=2.0E-4)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=2.0E-3)
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
    parser.add_argument('--envno', help='env NO.', default='2')
    parser.add_argument('--debug', help='store path', default='False')

    parser.set_defaults(render_env=False)
    # parser.set_defaults(render_env=True)

    # parser.set_defaults(use_gym_monitor=True)
    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(max_episodes=1.0E4)
    parser.set_defaults(max_episodes_len=1.0E5)
    # parser.set_defaults(minibatch_size=64)
    parser.set_defaults(minibatch_size=128)
    # parser.set_defaults(env='PathFollowing-v0')

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)

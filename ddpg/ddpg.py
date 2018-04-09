""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
# -- coding: utf-8 --
# import tensorflow as tf
import numpy as np
import sys
from gym import wrappers
import argparse
import pprint as pp
from pathfollowing_env_v2 import PathFollowingV2
from pathfollowing_env_v1 import PathFollowingV1
from pathfollowing_env_v3 import PathFollowingV3
import os
import random
from AGV_Model import AGV
import csv


# ===========================
#   Tensorflow Summary Ops
# ===========================


# ===========================
#   Agent Training
# ===========================
def pidControl(env):
    from PIDControl import PID
    # P, I, D = -0.001, 0.0, -0.1
    # P, I, D = 3, 0.005, 0.25
    P, I, D = 1, 0.05, 6
    pid = PID(P, I, D)
    pid.SetPoint = np.pi * 0.5

    P, I, D = 5.4, 0.05, 6
    pid2 = PID(P, I, D)
    pid2.SetPoint = 0

    s = env.reset()
    # env.car.Ir, env.car.w_mss, env.car.Ip1 = [13, 0.03, 0.03], [20,2.3,2.3], 10
    # env.car.setMess(1000)
    feedBack = s[env.historyLength * 2 - 1]
    print "====="+str(feedBack)+"====="
    # curr_error = -0.05

    setBeta = []
    for i in range(env.max_time):
        curr_error = s[env.historyLength -1]
        feedBack = s[env.historyLength * 2 - 1]

        # targetBeta = curr_error * 10.4
        pid2.update(-curr_error)
        targetBeta = pid2.output

        set_beta = np.pi * 0.5 - targetBeta
        setBeta.append(set_beta)


        pid.SetPoint = set_beta
        pid.update(feedBack)
        outPut = -pid.output

        # if outPut > AGV.MAX_ORIENTATION:
        #     outPut = AGV.MAX_ORIENTATION
        #     print 'max'
        # elif outPut < -AGV.MAX_ORIENTATION:
        #     outPut = -AGV.MAX_ORIENTATION
        #     print 'min'
        # else:
        #     print 'normal'

        # env.car.setSpeed(1)
        s, r, terminal, info = env.step(np.array([[outPut, 0]]))
        if terminal or i == env.max_time-1:

            moveStorex, moveStorey = info.get("moveStore")[0], info.get("moveStore")[1]
            wheelx, wheely = info.get("wheel")[0], info.get("wheel")[1]
            action_r, action_s = info.get("action")[0], info.get("action")[1]
            speed = info.get("speed")
            error_record, beta_record = info.get("error"), info.get("beta")

            no = str(0)

            with open('movePath' + no + '.csv', 'wb') as f:
                csv_writer = csv.writer(f)
                for x, y in zip(moveStorex, moveStorey):
                    csv_writer.writerow([x, y])

            with open('wheelPath' + no + '.csv', 'wb') as f:
                csv_writer = csv.writer(f)
                for x, y in zip(wheelx, wheely):
                    csv_writer.writerow([x, y])

            with open('action' + no + '.csv', 'wb') as f:
                csv_writer = csv.writer(f)
                for x, y in zip(action_r, action_s):
                    csv_writer.writerow([x, y])

            with open('error' + no + '.csv', 'wb') as f:
                csv_writer = csv.writer(f)
                for x in error_record:
                    csv_writer.writerow([x])

            with open('speed' + no + '.csv', 'wb') as f:
                csv_writer = csv.writer(f)
                for x in speed:
                    csv_writer.writerow([x])

            with open('beta' + no + '.csv', 'wb') as f:
                csv_writer = csv.writer(f)
                # for x in beta_record:
                for x in setBeta:
                    csv_writer.writerow([x])
            break

        else:
            curr_beta = s[env.historyLength*2-1]
            curr_error = s[env.historyLength -1]
            feedBack = curr_beta


def main(args):
    if args['envno'] == '2':
        env = PathFollowingV2()
    elif args['envno'] == '1':
        env = PathFollowingV1()
    elif args['envno'] == '3':
        env = PathFollowingV3()
    else:
        env = PathFollowingV3()
    env.seed(int(args['random_seed']))
    pidControl(env)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')


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
    # parser.set_defaults(render_env=True)

    # parser.set_defaults(use_gym_monitor=True)
    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(max_episodes=2.0E4)
    parser.set_defaults(max_episodes_len=1.0E6)
    # parser.set_defaults(minibatch_size=64)
    parser.set_defaults(minibatch_size=128)
    # parser.set_defaults(env='PathFollowing-v0')

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)

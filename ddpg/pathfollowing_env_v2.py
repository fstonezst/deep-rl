# -- coding: utf-8 --
import gym
import math
import numpy as np
from gym.utils import seeding
from gym import spaces
from AGV import AGV
import matplotlib.pyplot as plt

class PathFollowingV2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def _reset(self):
        self.car = AGV()
        self.totalError = 0
        self.car_position = 0
        self.time = 0
        self.error_sum=0
        self.record_buffer = [0, 0, 0, 0, 0, 0]

        self.error_abs_sum = 0
        self.action_sum = 0
        self.action_buffer = []
        self.lastAction = 0
        self.pathStore = []
        self.moveStorex = []
        self.moveStorey = []
        self.state = np.array([0, 0, 0, 0, 0, 0, float(self.car.q[3]), float(self.car.q[4])])
        self.movepath = []
        return np.array(self.state)


    def __init__(self):
        self.car = AGV()
        self.totalError = 0
        self.car_position = 0
        self.time = 0
        self.error_sum = 0
        self.record_buffer = [0, 0, 0, 0, 0, 0]

        self.error_abs_sum = 0
        self.action_sum = 0
        self.action_buffer = []
        self.lastAction =np.array([0,0])

        self.pathStore = []
        self.moveStorex, self.moveStorey = [], []
        self.action_r_store,self.action_s_store = [], []

        self.error_max_queue = []
        self.error_min_queue = []
        self.action_max_queue = []
        self.action_min_queue = []

        self.buffer_size = 10
        self.min_position = -1
        self.max_position = 1

        self.max_time = 800
        self.viewer = None

        self.path = [math.sin(x*(math.pi/180.0)) for x in range(0, 512)]
        self.r = 10
        self.movepath = [0 for _ in range(0, 180)]

        # action bounded
        min_orientation, max_orientation = -10 ,10
        min_rotation, max_rotation = 0, 600
        self.action_min = np.array([min_orientation, min_rotation])
        self.action_max = np.array([max_orientation, max_rotation])

        # observation bounded
        # x_max, x_min = 10, -10
        # y_max, y_min = 10, -10
        # xita_max, xita_min = np.pi, -np.pi
        B_max, B_min = np.pi, 0
        Error_max, Error_min = 2, -2
        speed_min, speed_max = 0, 50

        # self.observation_min = np.array([x_min, y_min, xita_min, B_min, Error_min, speed_min])
        # self.observation_max = np.array([x_max, y_max, xita_max, B_max, Error_max, speed_max])
        self.observation_min = np.array([B_min, Error_min, speed_min])
        self.observation_max = np.array([B_max, Error_max, speed_max])

        # space defined
        self.action_space = spaces.Box(self.action_min,self.action_max)
        self.observation_space = spaces.Box(self.observation_min, self.observation_max)

        self._seed()
        self._reset()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.car.controlInput(action)
        self.action_r_store.append(float(action[0][0]))
        self.action_s_store.append(float(action[0][1]))

        actionDiff = action - self.lastAction
        actionDiff = actionDiff[0]
        self.lastAction = action

        self.time += 1

        curcarx, curcary = float(self.car.q[0]), float(self.car.q[1])
        self.moveStorex.append(curcarx)
        self.moveStorey.append(curcary)

        # self.moveStore.append(self.car_position)
        error = (np.square(curcarx) + np.square(curcary)) - np.square(self.r)

        self.totalError += abs(error)
        self.record_buffer.append(error)
        self.error_sum += error
        self.error_abs_sum += abs(error)

        if len(self.record_buffer) > self.buffer_size:
            old = self.record_buffer.pop(0)
            self.error_sum -= old
            self.error_abs_sum -= abs(old)

        # error_min = min(self.error_abs_sum)
        # error_max = max(self.error_abs_sum)

        # action_min= min(self.action_buffer)

        # self.state = np.array([error, self.record_buffer[-2],  self.record_buffer[-3],\
        #                        self.record_buffer[-4], self.record_buffer[-5], self.record_buffer[-6]])

        # self.state = np.array(self.record_buffer)
        st = self.record_buffer[:6]
        B, speed = float(self.car.q[3]), float(self.car.q[4])
        st.extend([B,speed])
        # st = st.extend([float(self.car.q[3]), float(self.car.q[4])])
        self.state = np.array(st)

        ratio = 0.95
        reward = 0
        diff1, diff2 = actionDiff[0], actionDiff[1]
        if abs(error) > 0.03:
            reward += ratio * abs(error)
        if abs(diff1) > 0.05 :
            reward += (1-ratio) * abs(actionDiff[0])
        # if abs(actionDiff[1]) > 0.05 :
        if abs(diff2) > 0.05 :
            reward += (1-ratio) * abs(actionDiff[1])

        done = True if self.time > self.max_time or abs(error) > 100 else False

        if done:
            fig = plt.figure()
            ax = fig.add_subplot(2,1,1)
            ax1 = fig.add_subplot(2,1,2)
            ax.plot(self.moveStorex,self.moveStorey,'g-',label='move_path')
            ax1.plot(self.action_r_store,'r-',label='dir')
            ax1.plot(self.action_s_store,'b-',label='speed')
            plt.show()
            # print(self.totalError/self.time)
            # return np.array(self.state), -reward, done, {"result": [self.pathStore, self.moveStore], \
                                                     # "avgError": [self.totalError/float(self.time)]}
            return np.array(self.state), -reward, done, {"result": [], \
                                                         "avgError": [self.totalError/float(self.time)]}
        # return np.array(self.state), -reward, done, {"result": [self.pathStore, self.moveStore]}
        return np.array(self.state), -reward, done, {"result": []}





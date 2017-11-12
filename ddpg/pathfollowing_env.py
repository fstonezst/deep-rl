# -- coding: utf-8 --
import gym
import math
import numpy as np
from gym.utils import seeding
from gym import spaces


class PathFollowing(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    def _reset(self):
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
        self.moveStore = []
        self.state = np.array([0, 0, 0, 0, 0, 0])
        self.movepath = []
        return np.array(self.state)


    def __init__(self):
        self.totalError = 0
        self.car_position = 0
        self.time = 0
        self.error_sum = 0
        self.record_buffer = [0, 0, 0, 0, 0, 0]

        self.error_abs_sum = 0
        self.action_sum = 0
        self.action_buffer = []
        self.lastAction = 0

        self.pathStore = []
        self.moveStore = []

        self.error_max_queue = []
        self.error_min_queue = []
        self.action_max_queue = []
        self.action_min_queue = []


        self.buffer_size = 10
        self.min_position = -1
        self.max_position = 1
        self.min_action = -0.3
        self.max_action = 0.3
        self.max_time = 800

        self.viewer = None

        self.low = np.array([self.min_position])
        self.high = np.array([self.max_position])

        self.path = [math.sin(x*(math.pi/180.0)) for x in range(0, 512)]
        self.movepath = [0 for _ in range(0, 180)]

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1, 1))
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()
        self._reset()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.car_position += action[0]
        actionDiff = action[0] - self.lastAction
        self.lastAction = action[0]

        path_position = self.path[self.time & (len(self.path)-1)]
        self.time += 1

        self.pathStore.append(path_position)
        self.moveStore.append(self.car_position)

        error = self.car_position-path_position
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

        self.state = np.array([error, self.record_buffer[-2],  self.record_buffer[-3],\
                               self.record_buffer[-4], self.record_buffer[-5], self.record_buffer[-6]])
        ratio = 0.95
        reward = abs(error) * ratio + abs(actionDiff) * (1-ratio)

        done = True if self.time > self.max_time or abs(error) > 1 else False

        if done:
            print(self.totalError/self.time)
        return np.array(self.state), -reward, done, {"result": [self.pathStore, self.moveStore], "times": self.time}






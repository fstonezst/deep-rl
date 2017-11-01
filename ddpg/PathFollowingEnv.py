import gym
import math
import numpy as np
from gym.utils import seeding
from gym import spaces

class PathFollowing(gym.Env):

    def __init__(self):
        self.time = 0
        self.error_sum = 0
        self.record_buffer = [0]
        self.buffer_size = 10
        self.min_position = -1
        self.max_position = 1
        self.min_action = -0.5
        self.max_action = 0.5
        self.max_time = 5000

        self.low = np.array([self.min_position])
        self.high = np.array([self.max_position])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action)#Discrete(11)
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        car_position = self.state[0] + action

        path_position = math.sin(self.time)
        self.time += 3.14/180

        error = car_position-path_position
        self.record_buffer.append(error)
        self.error_sum += error

        if len(self.record_buffer) > self.buffer_size:
            old = self.record_buffer.pop(0)
            self.error_sum -= old
        error_i = self.error_sum/len(self.record_buffer)

        if len(self.record_buffer) > 1:
            error_d = self.record_buffer[-1] - self.record_buffer[-2]
        else:
            error_d = self.record_buffer[-1]
        self.state = np.array([error, error_i, error_d])
        reward = error * -1 + error_d * -0.1 + error_i * -0.01
        done = True if self.time > self.max_time or abs(car_position) > 2 else False
        return self.state, reward, done, {}

    def _reset(self):
        self.time = 0
        self.state = np.array([0, 0, 0])
        # self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)





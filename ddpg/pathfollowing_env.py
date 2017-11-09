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

    def __init__(self):
        self.totalError = 0
        self.car_position = 0
        self.time = 0
        self.error_sum = 0
        self.record_buffer = [0]
        self.lastAction = 0

        self.buffer_size = 10
        self.min_position = -1
        self.max_position = 1
        self.min_action = -0.3
        self.max_action = 0.3
        self.max_time = 1000

        self.viewer = None

        self.low = np.array([self.min_position])
        self.high = np.array([self.max_position])

        self.path = [math.sin(x*(math.pi/180.0)) for x in range(0, 360)]
        self.movepath = [0 for _ in range(0, 180)]

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1, 1))#Discrete(11)
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()
        self._reset()

        self.pathStore = []
        self.moveStore = []

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.car_position += action[0]
        actionDiff = action[0] - self.lastAction
        self.lastAction = action[0]
        # self.movepath.append(self.car_position)
        # self.movepath.pop(0)
        # path_position = math.sin(self.time)

        path_position = self.path[self.time % len(self.path)]
        self.time += 1
        # path_position = self.path[180]

        self.pathStore.append(path_position)
        self.moveStore.append(self.car_position)

        self.path.append(self.path.pop(0))

        # self.time += np.pi/float(180)
        # self.time = self.time + 1 if self.time+1 < len(self.path) else 0

        error = self.car_position-path_position
        self.totalError += abs(error)
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
        # reward = abs(error) * 1 + abs(error_d) * 0.1 + abs(error_i) * 0.01
        ratio = 0.8
        # reward = abs(error) * 1 + abs(actionDiff) * 0.2
        reward = abs(error) * ratio + abs(actionDiff) * (1-ratio)
        done = True if self.time > self.max_time or abs(error) > 2 else False
        # done = True if self.time > self.max_time else False
        # return np.array(self.state), 1/(reward+0.001), done, {}
        if done:
            print(self.totalError/self.time)
        return np.array(self.state), -reward, done, {"result":[self.pathStore,self.moveStore],"times":self.time}

    def _reset(self):
        self.lastAction = 0
        self.totalError = 0
        self.pathStore = []
        self.moveStore = []
        self.time = 0
        self.car_position = 0
        self.state = np.array([0, 0, 0])
        self.record_buffer = [0]
        self.path = [math.sin(x*(math.pi/180.0)) for x in range(0, 360)]
        self.movepath = []
        # self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)


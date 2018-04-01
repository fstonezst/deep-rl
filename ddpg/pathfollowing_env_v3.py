# -- coding: utf-8 --
import gym
import math
import numpy as np
from gym.utils import seeding
from gym import spaces
from AGV_Model import AGV
import random

class PathFollowingV3(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    # max_speed, min_speed = AGV.MAX_SPEED, 0
    # max_angle, min_angle = AGV.MAX_ANGLE, AGV.MIN_ANGLE
    # error_bound = 1
    # history_length = 4
    # leftOrRightTime = 0

    def _reset(self):
        # random = self.np_random
        history_len = self.historyLength
        if self.random_model:
            wheelx, wheely = random.randint(-3, 3) * 0.1 + 10, 0
            theta = random.randint(-35, 35) * 0.01 + np.pi
        else:
            # wheelx, wheely = random.randint(-10, 10) * 0.01 + 10, 0
            # theta = random.randint(-20, 20) * 0.01 + np.pi
            wheelx, wheely = 10, 0
            # wheelx, wheely = 9.9, 0
            theta = np.pi
        self.car = AGV(wheelPos=[wheelx, wheely], theta=theta, Ip1=self.Ip1)
        self.totalError = 0
        self.maxError = 0
        self.time = 0
        self.lastAction = 0

        #ERROR Cul
        self.startx, self.starty = 10, 0
        if self.random_model:
            self.left = random.randint(0, 1)
            self.firstLineLength, self.secondLineLength, self.midLineLength = random.randint(0,20) * 0.1, 10, \
                                                                          random.randint(20, 30) * 0.1
            self.firstArcR, self.secondArcR = random.choice([4, 6]), random.choice([4, 6])
        else:
            self.left = 1
            self.firstLineLength, self.secondLineLength, self.midLineLength = 4, 10, 4
            self.firstArcR, self.secondArcR = 6, 4

        if self.left:
            self.firstArcx, self.firstArcy = self.startx-self.firstArcR, self.starty+self.firstLineLength
            self.secondArcx, self.secondArcy = self.firstArcx-self.midLineLength, self.firstArcy+self.firstArcR+self.secondArcR
        else:
            self.firstArcx, self.firstArcy = self.startx+self.firstArcR, self.starty+self.firstLineLength
            self.secondArcx, self.secondArcy = self.firstArcx+self.midLineLength, self.firstArcy+self.firstArcR+self.secondArcR


        # self.lastAction = 0

        self.error_record, self.beta_record = [], []

        self.center_x_record, self.center_y_record = [], []
        self.wheel_x_record, self.wheel_y_record = [], []
        self.action_r_record, self.action_s_record = [], []
        self.speed_record = []
        # self.error_reward_record, self.speed_reward_record = [], []
        self.error_reward_record = []

        self.error_buffer = [0] * history_len
        self.beta_buffer = [0] * history_len
        self.u0_buffer = [0] * history_len
        self.u1_buffer = [0] * history_len

        # state
        errorState, betaState = [0] * history_len, [float(self.car.q[3])] * history_len
        u0State, u1State = [1] * history_len, [0] * history_len

        self.state = errorState + betaState + u0State + u1State
        return np.array(self.state)

    def __init__(self, hislen=50, error_bound=0.5, isRandom=False, Ip1 = 20):
        self.random_model = isRandom
        self.error_bound = error_bound
        self.max_time = 300
        self.viewer = None
        self.historyLength = hislen
        self.Ip1 = Ip1

        # self.car = AGV()
        self.totalError = 0
        self.maxError = 0
        self.time = 0
        self.lastAction = np.array([0, 0])

        # self.pathStore = []
        self.center_x_record, self.center_y_record = [], []
        self.wheel_x_record, self.wheel_y_record = [], []
        self.action_r_record, self.action_s_record = [], []
        self.speed_record = []
        # self.error_reward_record, self.speed_reward_record = [], []
        self.error_reward_record = []
        self.error_record, self.beta_record = [], []

        # self.buffer_size = 10

        # action bounded
        min_orientation, max_orientation = -AGV.MAX_ORIENTATION, AGV.MAX_ORIENTATION
        min_rotation, max_rotation = -AGV.MAX_ROTATION, AGV.MAX_ROTATION
        # self.action_min = np.array([min_orientation, min_rotation])
        # self.action_max = np.array([max_orientation, max_rotation])
        self.action_min = np.array([min_orientation])
        self.action_max = np.array([max_orientation])

        # observation bounded
        B_max, B_min = AGV.MAX_ANGLE, AGV.MIN_ANGLE
        Error_max, Error_min = self.error_bound, -self.error_bound
        speed_min, speed_max = 0, AGV.MAX_SPEED

        self.observation_min = np.array([B_min, Error_min, speed_min])
        self.observation_max = np.array([B_max, Error_max, speed_max])

        # space defined
        self.action_space = spaces.Box(self.action_min, self.action_max)
        self.observation_space = spaces.Box(self.observation_min, self.observation_max)

        self.seedNum = 1234
        self._seed(self.seedNum)
        self._reset()

    def setMaxTime(self, times):
        self.max_time=times
    def setCarMess(self, m):
        self.car.setMess(m)

    def _seed(self, seed=None):
        if seed is None:
            seed = self.seedNum
        self.np_random, self.seedNum = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        history_len = self.historyLength
        self.time += 1

        # control
        orientation = float(action[0][0])
        action = np.array([[orientation, 0]])
        # np.append(action[0], 0)
        self.car.controlInput(np.matrix(action))

        # AGV new state
        curcarx, curcary = float(self.car.q[0]), float(self.car.q[1])
        wheelx, wheely = float(self.car.wheelPos[0]), float(self.car.wheelPos[1])
        speed, orientationSpeed = float(self.car.uk[0]), float(self.car.uk[1])
        theta, beta = float(self.car.q[2]), float(self.car.q[3])


        self.u0_buffer.append(speed)
        self.u1_buffer.append(orientationSpeed)
        self.beta_buffer.append(beta)



        #ERROR Cul
        left = self.left
        startx, starty = self.startx, self.starty
        # firstLineLength, secondLineLength, midLineLength = self.firstLineLength, self.secondLineLength, self.midLineLength
        firstArcR, secondArcR = self.firstArcR, self.secondArcR
        firstArcx, firstArcy = self.firstArcx, self.firstArcy
        secondArcx, secondArcy = self.secondArcx, self.secondArcy

        refx, refy = wheelx, wheely
        record_x, record_y = curcarx, curcary
        # refx, refy = curcarx, curcary

        # 左正右负
        if left:
            if refy <= firstArcy:
                # first line
                referror = refx - startx
                record_error = record_x - startx
            elif refy >= secondArcy:
                # final line
                referror = refx - (secondArcx - secondArcR)
                record_error = record_x - (secondArcx - secondArcR)
            elif refx >= firstArcx:
                # first arc ni shi zhen
                referror = np.sqrt(np.square(refx - firstArcx) + np.square(refy - firstArcy)) - firstArcR
                record_error = np.sqrt(np.square(record_x - firstArcx) + np.square(record_y - firstArcy)) - firstArcR
            elif refx <= secondArcx:
                # second arc shun shi zhen
                referror = secondArcR - np.sqrt(np.square(refx - secondArcx) + np.square(refy - secondArcy))
                record_error = secondArcR - np.sqrt(np.square(record_x - secondArcx) + np.square(record_y - secondArcy))
            else:
                # mid line
                referror = refy - (firstArcy + firstArcR)
                record_error = record_y - (firstArcy + firstArcR)
        else:
            if refy <= firstArcy:
                referror = refx - startx
            elif refy >= secondArcy:
                referror = refx - (secondArcx + secondArcR)
            elif refx <= firstArcx:
                # shun shi zhen
                referror =  firstArcR - np.sqrt(np.square(refx - firstArcx) + np.square(refy - firstArcy))
            elif refx >= secondArcx:
                # ni shi zhen
                referror = np.sqrt(np.square(refx - secondArcx) + np.square(refy - secondArcy)) - secondArcR
            else:
                referror = refy - (firstArcy + firstArcR)

        if left:
            if record_y <= firstArcy:
                # first line
                record_error = record_x - startx
            elif record_y >= secondArcy:
                # final line
                record_error = record_x - (secondArcx - secondArcR)
            elif record_x >= firstArcx:
                # first arc ni shi zhen
                record_error = np.sqrt(np.square(record_x - firstArcx) + np.square(record_y - firstArcy)) - firstArcR
            elif record_x <= secondArcx:
                # second arc shun shi zhen
                record_error = secondArcR - np.sqrt(np.square(record_x - secondArcx) + np.square(record_y - secondArcy))
            else:
                # mid line
                record_error = record_y - (firstArcy + firstArcR)

        self.error_buffer.append(referror)


        if len(self.error_buffer) > history_len:
            self.error_buffer.pop(0)
            self.beta_buffer.pop(0)
            self.u0_buffer.pop(0)
            self.u1_buffer.pop(0)

        # State

        st = self.error_buffer + self.beta_buffer + self.u0_buffer + self.u1_buffer
        self.state = np.array(st)

        # Reward
        # error_reward = np.square(error) * 8.0E-1
        # speed_reward = 6.6E-3 / np.square(speed + 8.0E-2)   # 待测试
        # reward = speed_reward + error_reward
        orientation, rotation = float(action[0][0]), float(action[0][1])
        # reward = reward * 0.95 + 0.05 * (abs(orientation) / AGV.MAX_ORIENTATION)
        error_reward = np.square(referror) / np.square(self.error_bound)
        out_reward = abs(orientation) / AGV.MAX_ORIENTATION
        w = 0.99
        # w = 1
        reward = error_reward * w + (1 - w) * out_reward


        # Record
        self.action_r_record.append(orientation)
        self.action_s_record.append(rotation)
        # self.speed_reward_record.append(-speed_reward)
        self.error_reward_record.append(-error_reward)
        self.totalError += abs(referror)
        if abs(referror) > self.maxError:
            self.maxError = abs(referror)
        self.center_x_record.append(curcarx)
        self.center_y_record.append(curcary)
        self.wheel_x_record.append(wheelx)
        self.wheel_y_record.append(wheely)
        self.speed_record.append(speed)
        self.error_record.append(-referror)
        # self.error_record.append(record_error)
        self.beta_record.append(beta)

        # actionDiff = action - self.lastAction
        # self.lastAction = action
        # actionDiff = actionDiff[0]
        # diff1, diff2 = actionDiff[0], actionDiff[1]


        done = True if self.time > self.max_time or abs(referror) > self.error_bound else False
        # done = True if wheely >= yabound or abs(error) > PathFollowingV3.error_bound else False

        if done:
            return np.array(self.state), -reward, done, {"result": [], \
                                                         "avgError": self.maxError,
                                                         "moveStore": [self.center_x_record, self.center_y_record],
                                                         "action": [self.action_r_record, self.action_s_record],
                                                         "wheel": [self.wheel_x_record, self.wheel_y_record],
                                                         "speed": self.speed_record,
                                                         "error": self.error_record,
                                                         "beta": self.beta_record,
                                                         # "reward": [self.speed_reward_record, self.error_reward_record]}
                                                        "reward": [self.error_reward_record]}

        return np.array(self.state), -reward, done, {"result": []}
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

    max_speed, min_speed = AGV.MAX_SPEED, 0
    max_angle, min_angle = AGV.MAX_ANGLE, AGV.MIN_ANGLE
    error_bound = 1
    # history_length = 4
    leftOrRightTime = 0

    def _reset(self):
        # random = self.np_random
        history_len = self.historyLength
        if self.random_model:
            wheelx, wheely = random.randint(-3, 3) * 0.1 + 10, 0
            theta = random.randint(-35, 35) * 0.01 + np.pi
        else:
            wheelx, wheely = 10.1, 0
            theta = np.pi
        self.car = AGV(wheelPos=[wheelx, wheely], theta=theta)
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

        self.error_record = []

        self.center_x_record, self.center_y_record = [], []
        self.wheel_x_record, self.wheel_y_record = [], []
        self.action_r_record, self.action_s_record = [], []
        self.speed_record = []
        self.error_reward_record, self.speed_reward_record = [], []

        self.error_buffer = [0] * history_len
        self.beta_buffer = [0] * history_len
        self.u0_buffer = [0] * history_len
        self.u1_buffer = [0] * history_len

        # state
        errorState, betaState = [0] * history_len, [float(self.car.q[3])] * history_len
        u0State, u1State = [0] * history_len, [0] * history_len

        self.state = errorState + betaState + u0State + u1State
        return np.array(self.state)

    def __init__(self, hislen=2, isRandom=False):
        self.random_model = isRandom
        self.max_time = 20
        self.viewer = None
        self.historyLength = hislen

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
        self.error_reward_record, self.speed_reward_record = [], []
        self.error_record = []

        # self.buffer_size = 10

        # action bounded
        min_orientation, max_orientation = -AGV.MAX_ORIENTATION, AGV.MAX_ORIENTATION
        min_rotation, max_rotation = -AGV.MAX_ROTATION, AGV.MAX_ROTATION
        # self.action_min = np.array([min_orientation, min_rotation])
        # self.action_max = np.array([max_orientation, max_rotation])
        self.action_min = np.array([min_orientation])
        self.action_max = np.array([max_orientation])

        # observation bounded
        B_max, B_min = PathFollowingV3.max_angle, PathFollowingV3.min_angle
        Error_max, Error_min = PathFollowingV3.error_bound, -PathFollowingV3.error_bound
        speed_min, speed_max = PathFollowingV3.min_speed, PathFollowingV3.max_speed

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

        # 左正右负
        if left:
            if wheely <= firstArcy:
                # first line
                error = wheelx - startx
            elif wheely >= secondArcy:
                # final line
                error = wheelx - (secondArcx - secondArcR)
            elif wheelx >= firstArcx:
                # first arc ni shi zhen
                error = np.sqrt(np.square(wheelx - firstArcx) + np.square(wheely - firstArcy)) - firstArcR
            elif wheelx <= secondArcx:
                # second arc shun shi zhen
                error = secondArcR - np.sqrt(np.square(wheelx - secondArcx) + np.square(wheely - secondArcy))
            else:
                # mid line
                error = wheely - (firstArcy + firstArcR)
        else:
            if wheely <= firstArcy:
                error = wheelx - startx
            elif wheely >= secondArcy:
                error = wheelx - (secondArcx + secondArcR)
            elif wheelx <= firstArcx:
                # shun shi zhen
                error =  firstArcR - np.sqrt(np.square(wheelx - firstArcx) + np.square(wheely - firstArcy))
            elif wheelx >= secondArcx:
                # ni shi zhen
                error = np.sqrt(np.square(wheelx - secondArcx) + np.square(wheely - secondArcy)) - secondArcR
            else:
                error = wheely - (firstArcy + firstArcR)

        self.error_buffer.append(error)


        if len(self.error_buffer) > history_len:
            self.error_buffer.pop(0)
            self.beta_buffer.pop(0)
            self.u0_buffer.pop(0)
            self.u1_buffer.pop(0)

        # State

        st = self.error_buffer + self.beta_buffer + self.u0_buffer + self.u1_buffer
        self.state = np.array(st)

        # Reward
        error_reward = np.square(error) * 8.0E-1
        speed_reward = 6.6E-3 / np.square(speed + 8.0E-2)   # 待测试
        reward = speed_reward + error_reward
        orientation, rotation = float(action[0][0]), float(action[0][1])
        # reward = reward * 0.95 + 0.05 * (abs(orientation) / AGV.MAX_ORIENTATION)
        reward = reward * 0.9995 + 0.0005 * (abs(orientation) / AGV.MAX_ORIENTATION)


        # Record
        self.action_r_record.append(orientation)
        self.action_s_record.append(rotation)
        self.speed_reward_record.append(-speed_reward)
        self.error_reward_record.append(-error_reward)
        self.totalError += abs(error)
        if abs(error) > self.maxError:
            self.maxError = abs(error)
        self.center_x_record.append(curcarx)
        self.center_y_record.append(curcary)
        self.wheel_x_record.append(wheelx)
        self.wheel_y_record.append(wheely)
        self.speed_record.append(speed)
        self.error_record.append(error)

        # actionDiff = action - self.lastAction
        # self.lastAction = action
        # actionDiff = actionDiff[0]
        # diff1, diff2 = actionDiff[0], actionDiff[1]


        done = True if self.time > self.max_time or abs(error) > PathFollowingV3.error_bound else False
        # done = True if wheely >= yabound or abs(error) > PathFollowingV3.error_bound else False

        if done:
            return np.array(self.state), -reward, done, {"result": [], \
                                                         "avgError": self.maxError,
                                                         "moveStore": [self.center_x_record, self.center_y_record],
                                                         "action": [self.action_r_record, self.action_s_record],
                                                         "wheel": [self.wheel_x_record, self.wheel_y_record],
                                                         "speed": self.speed_record,
                                                         "error": self.error_record,
                                                         "reward": [self.speed_reward_record, self.error_reward_record]}

        return np.array(self.state), -reward, done, {"result": []}
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
    history_length = 4
    leftOrRightTime = 0

    def _reset(self):
        # random = self.np_random
        history_len = 4
        wheelx, wheely = random.randint(-9, 9) * 0.01 + 10, 0
        theta = random.randint(-35, 35) * 0.01 + np.pi
        self.car = AGV(wheelPos=[wheelx, wheely], theta=theta)
        self.totalError = 0
        self.time = 0
        self.lastAction = 0

        #ERROR Cul
        self.left = random.randint(0,1)

        self.startx, self.starty = 10, 0
        self.firstLineLength, self.secondLineLength, self.midLineLength = random.randint(0, 20)*0.1, 10, random.randint(30, 50)*0.1
        self.firstArcR, self.secondArcR = random.randint(30, 70) * 0.1, random.randint(30,70) * 0.1
        if self.left:
            self.firstArcx, self.firstArcy = self.startx-self.firstArcR, self.starty+self.firstLineLength
            self.secondArcx, self.secondArcy = self.firstArcx-self.midLineLength, self.firstArcy+self.firstArcR+self.secondArcR
        else:
            self.firstArcx, self.firstArcy = self.startx+self.firstArcR, self.starty+self.firstLineLength
            self.secondArcx, self.secondArcy = self.firstArcx+self.midLineLength, self.firstArcy+self.firstArcR+self.secondArcR



        self.moveStorex, self.moveStorey = [], []
        self.wheelx, self.wheely = [], []
        self.action_r_store, self.action_s_store = [], []
        self.speed = []
        self.error_reward_record, self.speed_reward_record = [], []

        self.error_record_buffer = [0] * history_len
        self.theta_record_buffer = [0] * history_len
        self.B_record_buffer = [0] * history_len
        self.u0_record_buffer = [0] * history_len
        self.u1_record_buffer = [0] * history_len

        self.derror_record_buffer = [0] * history_len
        self.dtheta_record_buffer = [0] * history_len
        self.dB_record_buffer = [0] * history_len
        self.du0_record_buffer = [0] * history_len
        self.du1_record_buffer = [0] * history_len



        errorState, u0State, u1State,  = [0] * history_len, [0] * history_len, [0] * history_len
        thetaState, BState= [float(self.car.q[2])] * history_len, [float(self.car.q[3])] * history_len
        derrorState, dthetaState, dBState = [0] * history_len, [0] * history_len, [0] * history_len
        du0State, du1State = [0] * history_len, [0] * history_len

        self.state = errorState + thetaState + BState + u0State + u1State \
            + derrorState + dthetaState + dBState + du0State + du1State
        return np.array(self.state)

    def __init__(self):
        self.buffer_size = 10
        self.max_time = 600
        self.viewer = None

        self.car = AGV()
        self.totalError = 0
        self.time = 0
        self.lastAction = np.array([0, 0])

        # self.pathStore = []
        self.moveStorex, self.moveStorey = [], []
        self.wheelx, self.wheely = [], []
        self.action_r_store, self.action_s_store = [], []
        self.speed = []
        self.error_reward_record, self.speed_reward_record = [], []

        # self.r = 10

        # action bounded
        min_orientation, max_orientation = -AGV.MAX_ORIENTATION, AGV.MAX_ORIENTATION
        min_rotation, max_rotation = -AGV.MAX_ROTATION, AGV.MAX_ROTATION
        self.action_min = np.array([min_orientation, min_rotation])
        self.action_max = np.array([max_orientation, max_rotation])

        # observation bounded
        # x_max, x_min = 10, -10
        # y_max, y_min = 10, -10
        # xita_max, xita_min = np.pi, -np.pi
        B_max, B_min = PathFollowingV3.max_angle, PathFollowingV3.min_angle
        Error_max, Error_min = PathFollowingV3.error_bound, -PathFollowingV3.error_bound
        speed_min, speed_max = PathFollowingV3.min_speed, PathFollowingV3.max_speed

        # self.observation_min = np.array([x_min, y_min, xita_min, B_min, Error_min, speed_min])
        # self.observation_max = np.array([x_max, y_max, xita_max, B_max, Error_max, speed_max])
        self.observation_min = np.array([B_min, Error_min, speed_min])
        self.observation_max = np.array([B_max, Error_max, speed_max])

        # space defined
        self.action_space = spaces.Box(self.action_min, self.action_max)
        self.observation_space = spaces.Box(self.observation_min, self.observation_max)

        self.seedNum = 1234
        self._seed(self.seedNum)
        self._reset()

    def setCarMess(self, m):
        self.car.setMess(m)

    def _seed(self, seed=None):
        if seed is None:
            seed = self.seedNum
        self.np_random, self.seedNum = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        history_len = PathFollowingV3.history_length
        self.time += 1

        # control
        self.car.controlInput(np.matrix(action))

        # AGV new state
        curcarx, curcary = float(self.car.q[0]), float(self.car.q[1])
        wheelx, wheely = float(self.car.wheelPos[0]), float(self.car.wheelPos[1])
        theta, B, speed = float(self.car.q[2]), float(self.car.q[3]), float(self.car.uk[0])  # float(self.car.q[4])

        #ERROR Cul
        left = self.left
        startx, starty = self.startx, self.starty
        firstLineLength, secondLineLength, midLineLength = self.firstLineLength, self.secondLineLength, self.midLineLength
        firstArcR, secondArcR = self.firstArcR, self.secondArcR
        firstArcx, firstArcy = self.firstArcx, self.firstArcy
        secondArcx, secondArcy = self.secondArcx, self.secondArcy
        # yabound = starty + firstLineLength + firstArcR + secondArcR + secondLineLength

        if left:
            if wheely <= firstArcy:
                # first line
                error = wheelx - startx
            elif wheely >= secondArcy:
                # final line
                error = wheelx - (secondArcx - secondArcR)
            elif wheelx >= firstArcx:
                # first arc
                error = np.sqrt(np.square(wheelx - firstArcx) + np.square(wheely - firstArcy)) - firstArcR
            elif wheelx <= secondArcx:
                # second arc
                error = np.sqrt(np.square(wheelx - secondArcx) + np.square(wheely - secondArcy)) - secondArcR
            else:
                # mid line
                error = wheely - (firstArcy + firstArcR)
        else:
            if wheely <= firstArcy:
                error = wheelx - startx
            elif wheely >= secondArcy:
                error = wheelx - (secondArcx + secondArcR)
            elif wheelx <= firstArcx:
                error = np.sqrt(np.square(wheelx - firstArcx) + np.square(wheely - firstArcy)) - firstArcR
            elif wheelx >= secondArcx:
                error = np.sqrt(np.square(wheelx - secondArcx) + np.square(wheely - secondArcy)) - secondArcR
            else:
                error = wheely - (firstArcy + firstArcR)


        # State Record
        self.error_record_buffer.append(error)
        self.theta_record_buffer.append(speed)
        self.B_record_buffer.append(B)
        self.u0_record_buffer.append(float(self.car.uk[0]))
        self.u1_record_buffer.append(float(self.car.uk[1]))

        self.derror_record_buffer.append(error-self.error_record_buffer[-1])
        self.dtheta_record_buffer.append(theta-self.theta_record_buffer[-1])
        self.dB_record_buffer.append(B-self.B_record_buffer[-1])
        self.du0_record_buffer.append(float(self.car.uk[0]) - self.u0_record_buffer[-1])
        self.du1_record_buffer.append(float(self.car.uk[1]) - self.u1_record_buffer[-1])


        if len(self.error_record_buffer) > self.buffer_size:
            self.error_record_buffer.pop(0)
            self.theta_record_buffer.pop(0)
            self.B_record_buffer.pop(0)
            self.u0_record_buffer.pop(0)
            self.u1_record_buffer.pop(0)

            self.derror_record_buffer.pop(0)
            self.dtheta_record_buffer.pop(0)
            self.dB_record_buffer.pop(0)
            self.du0_record_buffer.pop(0)
            self.du1_record_buffer.pop(0)

        # State
        error_state = self.error_record_buffer[-PathFollowingV3.history_length:]
        thetaState = self.theta_record_buffer[-history_len:]
        BState = self.B_record_buffer[-history_len:]
        u0_state = self.u0_record_buffer[-PathFollowingV3.history_length:]
        u1_state = self.u1_record_buffer[-PathFollowingV3.history_length:]

        derror_state = self.derror_record_buffer[-history_len:]
        dthata_state = self.dtheta_record_buffer[-history_len:]
        dB_state = self.dB_record_buffer[-history_len:]
        du0_state = self.du0_record_buffer[-history_len:]
        du1_state = self.du1_record_buffer[-history_len:]

        st = error_state + thetaState + BState + u0_state + u1_state\
             + derror_state + dthata_state + dB_state + du0_state + du1_state
        self.state = np.array(st)


        # Reward
        error_reward = np.square(error) * 8.0E-1
        speed_reward = 6.6E-3 / np.square(speed + 8.0E-2)   # 待测试
        reward = speed_reward + error_reward



        # Record
        orientation, rotation = float(action[0][0]), float(action[0][1])
        self.action_r_store.append(orientation)
        self.action_s_store.append(rotation)
        self.speed_reward_record.append(-speed_reward)
        self.error_reward_record.append(-error_reward)
        self.totalError += abs(error)
        self.moveStorex.append(curcarx)
        self.moveStorey.append(curcary)
        self.wheelx.append(wheelx)
        self.wheely.append(wheely)
        self.speed.append(speed)

        actionDiff = action - self.lastAction
        self.lastAction = action
        actionDiff = actionDiff[0]
        diff1, diff2 = actionDiff[0], actionDiff[1]


        done = True if self.time > self.max_time or abs(error) > PathFollowingV3.error_bound else False
        # done = True if wheely >= yabound or abs(error) > PathFollowingV3.error_bound else False

        if done:
            return np.array(self.state), -reward, done, {"result": [], \
                                                         "avgError": self.totalError / float(self.time),
                                                         "moveStore": [self.moveStorex, self.moveStorey],
                                                         "action": [self.action_r_store, self.action_s_store],
                                                         "wheel": [self.wheelx, self.wheely],
                                                         "speed": self.speed,
                                                         "reward": [self.speed_reward_record, self.error_reward_record]}

        return np.array(self.state), -reward, done, {"result": []}
# -- coding: utf-8 --
import gym
import math
import numpy as np
from gym.utils import seeding
from gym import spaces
from AGV_Model import AGV

class PathFollowingV2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    max_speed, min_speed = AGV.MAX_SPEED, 0
    max_angle, min_angle = AGV.MAX_ANGLE, AGV.MIN_ANGLE

    def _reset(self):
        self.car = AGV([self.r,0], np.pi)
        self.totalError = 0
        self.maxError = 0
        self.time = 0

        hislength = self.historyLength
        self.error_buffer = [0] * hislength
        self.beta_buffer = [0] * hislength
        # self.theta_buffer = [0] * hislength
        self.u0_buffer = [0] * hislength
        self.u1_buffer = [0] * hislength


        # self.lastAction = 0

        self.center_x_record, self.center_y_record = [], []
        self.wheel_x_record, self.wheel_y_record = [], []
        self.action_r_record, self.action_s_record = [], [] #[0] * hislength, [0] * hislength
        self.speed_record = []
        self.error_record = []

        # errorState, betaState, thetaState = [0] * hislength, [float(self.car.q[3])] * hislength, [float(self.car.q[2])] * hislength
        errorState, betaState = [0] * hislength, [float(self.car.q[3])] * hislength
        u0State, u1State = [0] * hislength, [0] * hislength
        # actionrState, actionoState = [0] * (hislength-1), [0] * (hislength-1)

        # self.state = errorState + betaState + thetaState + u0State + u1State
        self.state = errorState + betaState + u0State + u1State # + actionoState + actionrState

        return np.array(self.state)

    def __init__(self, max_time=1000, errorBound=1, r=5, hislength=3):
        self.car = AGV([r, 0], np.pi)
        self.totalError = 0
        self.maxError = 0
        self.time = 0

        self.historyLength = hislength
        self.error_buffer = [0] * hislength
        self.beta_buffer = [0] * hislength
        # self.theta_buffer = [0] * hislength
        self.u0_buffer = [0] * hislength
        self.u1_buffer = [0] * hislength


        # self.lastAction = np.array([0, 0])

        self.center_x_record, self.center_y_record = [], []
        self.wheel_x_record, self.wheel_y_record = [], []
        self.action_r_record, self.action_s_record = [], []
        self.speed_record = []
        self.error_record, self.speed_reward_record = [], []

        # self.buffer_size = 10

        self.max_time = max_time
        self.error_bound = errorBound
        self.viewer = None

        self.r = r

        # action bounded
        min_orientation, max_orientation = -AGV.MAX_ORIENTATION, AGV.MAX_ORIENTATION
        min_rotation, max_rotation = -AGV.MAX_ROTATION, AGV.MAX_ROTATION
        self.action_min = np.array([min_orientation, min_rotation])
        self.action_max = np.array([max_orientation, max_rotation])

        # observation bounded
        B_max, B_min = PathFollowingV2.max_angle, PathFollowingV2.min_angle
        Error_max, Error_min = self.error_bound, -self.error_bound
        speed_min, speed_max = PathFollowingV2.min_speed, PathFollowingV2.max_speed

        self.observation_min = np.array([B_min, Error_min, speed_min])
        self.observation_max = np.array([B_max, Error_max, speed_max])

        # space defined
        self.action_space = spaces.Box(self.action_min, self.action_max)
        self.observation_space = spaces.Box(self.observation_min, self.observation_max)

        self._seed()
        self._reset()

    def setCarMess(self, m):
        self.car.setMess(m)

    def setMaxTime(self, maxTime):
        self.max_time = maxTime

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.car.controlInput(np.matrix(action))
        orientation, rotation = float(action[0][0]), float(action[0][1])
        self.action_r_record.append(orientation)
        self.action_s_record.append(rotation)

        # actionDiff = action - self.lastAction
        # actionDiff = actionDiff[0]
        # self.lastAction = action

        self.time += 1


        curcarx, curcary = float(self.car.q[0]), float(self.car.q[1])
        wheelx, wheely = float(self.car.wheelPos[0]), float(self.car.wheelPos[1])
        speed, orientationSpeed = float(self.car.uk[0]), float(self.car.uk[1])
        theta, beta = float(self.car.q[2]), float(self.car.q[3])

        self.center_x_record.append(curcarx)
        self.center_y_record.append(curcary)
        self.wheel_x_record.append(wheelx)
        self.wheel_y_record.append(wheely)
        self.speed_record.append(speed)

        self.u0_buffer.append(speed)
        self.u1_buffer.append(orientationSpeed)
        # self.theta_buffer.append(theta)
        self.beta_buffer.append(beta)


        error = np.sqrt((np.square(wheelx) + np.square(wheely))) - self.r

        self.error_record.append(error)
        self.totalError += abs(error)
        if abs(error) > self.maxError:
            self.maxError = abs(error)

        self.error_buffer.append(error)
        hislen = self.historyLength
        if len(self.error_buffer) > hislen:
            self.error_buffer.pop(0)
            # self.theta_buffer.pop(0)
            self.beta_buffer.pop(0)
            self.u0_buffer.pop(0)
            self.u1_buffer.pop(0)

        st = self.error_buffer + self.beta_buffer + self.u0_buffer + self.u1_buffer #\
             # + self.action_r_record[-hislen:-2] + self.action_s_record[-hislen:-2]
        self.state = np.array(st)


        error_reward = np.square(error) * 8.0E-1
        speed_reward = 6.6E-3 / np.square(speed + 8.0E-2)

        reward = speed_reward + error_reward


        done = True if self.time > self.max_time or abs(error) > self.error_bound else False

        if done:
            return np.array(self.state), -reward, done, {"result": [], \
                                                         # "avgError": self.totalError / float(self.time),
                                                         # "avgError": self.maxError,
                                                         "avgError": abs(error),
                                                         "moveStore": [self.center_x_record, self.center_y_record],
                                                         "action": [self.action_r_record, self.action_s_record],
                                                         "wheel": [self.wheel_x_record, self.wheel_y_record],
                                                         "speed": self.speed_record,
                                                         "error": self.error_record}

        return np.array(self.state), -reward, done, {"result": [],
                                                     "Error": self.maxError
                                                     }

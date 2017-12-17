# -- coding: utf-8 --
import gym
import math
import numpy as np
from gym.utils import seeding
from gym import spaces
from AGV_Model import AGV


# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle

class PathFollowingV2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    max_speed, min_speed = AGV.MAX_SPEED, 0
    max_angle, min_angle = AGV.MAX_ANGLE, AGV.MIN_ANGLE
    error_bound = 2
    history_length = 6

    def _reset(self):
        self.car = AGV()
        self.totalError = 0
        self.time = 0
        self.error_record_buffer = [0] * PathFollowingV2.history_length
        self.u0_record_buffer = [0] * PathFollowingV2.history_length
        self.u1_record_buffer = [0] * PathFollowingV2.history_length

        self.lastAction = 0

        self.moveStorex, self.moveStorey = [], []
        self.wheelx, self.wheely = [], []
        self.action_r_store, self.action_s_store = [], []
        self.speed = []
        self.error_reward_record, self.speed_reward_record = [], []

        errorState, u0State, u1State = [0] * 6, [0] * 6, [0] * 6
        self.state = errorState + u0State + u1State
        B, speed = float(self.car.q[2]), float(self.car.q[3])
        self.state.extend([B, speed])

        return np.array(self.state)

    def __init__(self):
        self.car = AGV()
        self.totalError = 0
        self.time = 0

        self.error_record_buffer = [0] * PathFollowingV2.history_length
        self.u0_record_buffer = [0] * PathFollowingV2.history_length
        self.u1_record_buffer = [0] * PathFollowingV2.history_length

        self.lastAction = np.array([0, 0])

        # self.pathStore = []
        self.moveStorex, self.moveStorey = [], []
        self.wheelx, self.wheely = [], []
        self.action_r_store, self.action_s_store = [], []
        self.speed = []
        self.error_reward_record, self.speed_reward_record = [], []

        # self.error_max_queue = []
        # self.error_min_queue = []
        # self.action_max_queue = []
        # self.action_min_queue = []

        self.buffer_size = 10
        # self.min_position = -1
        # self.max_position = 1

        self.max_time = 300
        self.viewer = None

        # self.path = [math.sin(x*(math.pi/180.0)) for x in range(0, 512)]
        self.r = 10
        # self.movepath = [0 for _ in range(0, 180)]

        # action bounded
        min_orientation, max_orientation = -AGV.MAX_ORIENTATION, AGV.MAX_ORIENTATION
        min_rotation, max_rotation = -AGV.MAX_ROTATION, AGV.MAX_ROTATION
        self.action_min = np.array([min_orientation, min_rotation])
        self.action_max = np.array([max_orientation, max_rotation])

        # observation bounded
        # x_max, x_min = 10, -10
        # y_max, y_min = 10, -10
        # xita_max, xita_min = np.pi, -np.pi
        B_max, B_min = PathFollowingV2.max_angle, PathFollowingV2.min_angle
        Error_max, Error_min = PathFollowingV2.error_bound, -PathFollowingV2.error_bound
        speed_min, speed_max = PathFollowingV2.min_speed, PathFollowingV2.max_speed

        # self.observation_min = np.array([x_min, y_min, xita_min, B_min, Error_min, speed_min])
        # self.observation_max = np.array([x_max, y_max, xita_max, B_max, Error_max, speed_max])
        self.observation_min = np.array([B_min, Error_min, speed_min])
        self.observation_max = np.array([B_max, Error_max, speed_max])

        # space defined
        self.action_space = spaces.Box(self.action_min, self.action_max)
        self.observation_space = spaces.Box(self.observation_min, self.observation_max)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.car.controlInput(np.matrix(action))
        orientation, rotation = float(action[0][0]), float(action[0][1])
        self.action_r_store.append(orientation)
        self.action_s_store.append(rotation)

        actionDiff = action - self.lastAction
        actionDiff = actionDiff[0]
        self.lastAction = action

        self.time += 1

        curcarx, curcary = float(self.car.q[0]), float(self.car.q[1])
        wheelx, wheely = float(self.car.wheelPos[0]), float(self.car.wheelPos[1])
        self.moveStorex.append(curcarx)
        self.moveStorey.append(curcary)
        self.wheelx.append(wheelx)
        self.wheely.append(wheely)

        self.u0_record_buffer.append(float(self.car.uk[0]))
        self.u1_record_buffer.append(float(self.car.uk[1]))

        # error = (np.square(curcarx) + np.square(curcary)) - np.square(self.r)
        error = (np.square(wheelx) + np.square(wheely)) - np.square(self.r)

        self.totalError += abs(error)
        self.error_record_buffer.append(error)
        # self.error_sum += error
        # self.error_abs_sum += abs(error)

        if len(self.error_record_buffer) > self.buffer_size:
            self.error_record_buffer.pop(0)
            self.u0_record_buffer.pop(0)
            self.u1_record_buffer.pop(0)

        # error_min = min(self.error_abs_sum)
        # error_max = max(self.error_abs_sum)
        # action_min= min(self.action_buffer)

        error_state = self.error_record_buffer[-PathFollowingV2.history_length:]
        u0_state = self.u0_record_buffer[-PathFollowingV2.history_length:]
        u1_state = self.u1_record_buffer[-PathFollowingV2.history_length:]
        st = error_state + u0_state + u1_state
        theta, B, speed = float(self.car.q[2]), float(self.car.q[3]), float(self.car.uk[0])  # float(self.car.q[4])
        self.speed.append(speed)
        st.extend([theta, B])
        self.state = np.array(st)

        diff1, diff2 = actionDiff[0], actionDiff[1]

        # error_reward = np.square(error) * 50
        # error_reward = np.square(error * 3) * 2.0E2
        error_reward = np.square(error * 3) * 1.0E2


        speed_reward = 2.0E1 / (np.square(speed) + 5.0E-3) - 50 # 待测试
        # speed_reward = -np.log(speed + 1.0E-1) * 5.0E2
        if speed_reward < 0:
            speed_reward = 0


        reward = speed_reward + error_reward
        # reward /= 5000.0
        self.speed_reward_record.append(-speed_reward)
        self.error_reward_record.append(-error_reward)

        # if abs(diff1) > 0.01:
        #     reward += (1-ratio) * abs(actionDiff[0])
        # if abs(diff2) > 100:
        #     reward += (1-ratio) * abs(actionDiff[1])

        done = True if self.time > self.max_time or abs(error) > PathFollowingV2.error_bound else False

        if done:
            return np.array(self.state), -reward, done, {"result": [], \
                                                         "avgError": self.totalError / float(self.time),
                                                         "moveStore": [self.moveStorex, self.moveStorey],
                                                         "action": [self.action_r_store, self.action_s_store],
                                                         "wheel": [self.wheelx, self.wheely],
                                                         "speed": self.speed,
                                                         "reward": [self.speed_reward_record, self.error_reward_record]}

        return np.array(self.state), -reward, done, {"result": []}

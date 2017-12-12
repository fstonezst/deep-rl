import matplotlib.pyplot as plt
from AGV_Model import AGV
from pathfollowing_env_v2 import PathFollowingV2
import numpy as np
def error_fun(error):
    error_reward = np.square(error) * 50
    return error_reward
def speed_fun(speed):
    # speed_reward = -np.log(speed + 5.0E-30) * 1.0E2
    # 2 * np.square(speed)
    speed_reward = 1.0E2 / (np.square(speed) + 3.0E-2) - 50
    return speed_reward
    # return 0.1 / (np.square(idddpg/show_reward_fun.py:13dpg/show_reward_fun.py:13) + 0.001)
    # return 1 / (i + 0.005)ddpg/show_reward_fun.py:13

step, start, end = 0.01, 0, 5 #AGV.MAX_SPEED
scalEnd = int(end * (1 / step))
list = [error_fun(i * step) for i in range(0, scalEnd)]

step, start, end = 0.01, 0, 1 #PathFollowingV2.error_bound
scalEnd = int(end * (1 / step))
list2 = [speed_fun(i * step) for i in range(0, scalEnd)]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# ax.plot(list,'r-')
ax.plot(list2,'b-')
plt.show()

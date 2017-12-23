# -- coding: utf-8 --
import matplotlib.pyplot as plt
from AGV_Model import AGV
from pathfollowing_env_v2 import PathFollowingV2
import numpy as np
def error_fun(error):
    error_reward = np.square(error) * 6.0E-1
    # error_reward = np.square(error) * 1.0E0
    return error_reward
    # return 0
def speed_fun(speed):
    # speed_reward = 1.0E-1 / (speed + 1.0E-1)   # 待测试

    speed_reward = 4E-2 / (speed + 4.0E-2)  # 待测试
    # speed_reward = 6.6E-3 / np.square(speed + 8.0E-2)   # 待测试
    return speed_reward
    # return 0

# step, start, end = 0.01, 0, 9 #AGV.MAX_SPEED
# scalEnd = int(end * (1 / step))
# list = [error_fun(i * step) for i in range(0, scalEnd)]
#
# step, start, end = 0.01, 0, 2 #PathFollowingV2.error_bound
# scalEnd = int(end * (1 / step))
# list2 = [speed_fun(i * step) for i in range(0, scalEnd)]
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(list,'r-')
# ax.plot(list2,'b-')
# plt.show()


from mpl_toolkits.mplot3d import Axes3D

# step, start, end = 0.05, 0, 1 #AGV.MAX_SPEED
step, start, end = 0.1, 0.0, 1 #AGV.MAX_SPEED
X = np.arange(start, end, step)
step, start, end = 0.1, 0.0, 1 #PathFollowingV2.error_bound
Y = np.arange(start, end, step)
X, Y = np.meshgrid(X, Y)
Z = np.matrix(map(error_fun, X)) + np.matrix(map(speed_fun, Y))
Z = np.array(Z)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
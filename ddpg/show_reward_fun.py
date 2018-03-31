# -- coding: utf-8 --
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def error_fun(error):
    # error_reward = np.square(error) * 8.0E-1
    error_reward = np.square(error) / np.square(0.5)

    #导函数
    derivative = 1.6E0 * error
    return error_reward * 0.99
    # return derivative
    # return 0

def speed_fun(speed):
    speed_reward = 6.6E-3 / np.square(speed + 8.0E-2)   # 待测试

    #导函数
    derivative = -1.32E-2 / np.power(speed + 8.0E-2, 3)
    return speed_reward
    # return derivative
    # return 0
def out_fun(out):
    return (out / 0.5) * 0.01


# step, start, end = 0.01, 0.0, 0.1  #PathFollowingV2.error_bound
step, start, end = 0.005, 0.0, 0.05  #PathFollowingV2.error_bound
X = np.arange(start, end, step)
step, start, end = 0.05, 0, 0.5         #AGV.MAX_SPEED
Y = np.arange(start, end, step)
X, Y = np.meshgrid(X, Y)

# total reward
# Z = np.matrix(map(error_fun, X)) + np.matrix(map(speed_fun, Y))
Z = np.matrix(map(error_fun, X)) + np.matrix(map(out_fun, Y))
#speed reward proportion
# Z = np.matrix(map(speed_fun, Y))/(np.matrix(map(error_fun, X)) + np.matrix(map(speed_fun, Y)))
# diff
# Z = np.matrix(map(error_fun, X)) - np.matrix(map(speed_fun, Y))

Z = np.array(Z)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
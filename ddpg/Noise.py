# -- coding: utf-8 --
# ===========================
#   Noise - Ornstein Uhlenbeck
#   Modified from: http://www.turingfinance.com/random-walks-down-wall-street-stochastic-processes-in-python/
#   Author: Liam Pettigrew
# ===========================
import numpy as np

class OrnsteinUhlenbeckNoise(object):


    def __init__(self, delta=0.5, sigma=0.5, ou_a=3.0, ou_mu=0.0):
        # Noise parameters
        # DELTA = 0.1 # The rate of change (time)
        # SIGMA = 5 # Volatility of the stochastic processes
        # OU_A = 3. # The rate of mean reversion
        # OU_MU = 0. # The long run average interest rate
        self.delta = delta
        self.sigma = sigma
        self.ou_a = ou_a
        self.ou_mu = ou_mu

    def brownian_motion_log_returns(self):
        """
        This method returns a Wiener process. The Wiener process is also called Brownian motion. For more information
        about the Wiener process check out the Wikipedia page: http://en.wikipedia.org/wiki/Wiener_process
        :return: brownian motion log returns
        """
        sqrt_delta_sigma = np.sqrt(self.delta) * self.sigma
        return np.random.normal(loc=0, scale=sqrt_delta_sigma, size=None)

    def ornstein_uhlenbeck_level(self, prev_ou_level):
        """
        This method returns the rate levels of a mean-reverting ornstein uhlenbeck process.
        :return: the Ornstein Uhlenbeck level
        """
        drift = self.ou_a * (self.ou_mu - prev_ou_level) * self.delta
        randomness = self.brownian_motion_log_returns()
        return prev_ou_level + drift + randomness

# import random
# # Noise parameters - Ornstein Uhlenbeck
# import matplotlib.pyplot as plt
# DELTA = 0.1 # The rate of change (time)
# SIGMA = 5 # Volatility of the stochastic processes
# OU_A = 3. # The rate of mean reversion
# OU_MU = 0. # The long run average interest rate
#
# ou_level = 0.
# noise = OrnsteinUhlenbeckNoise(DELTA, SIGMA, OU_A, OU_MU)
# record = []
# random_record = []
#
# for _ in range(100):
#     ou_level = noise.ornstein_uhlenbeck_level(ou_level)
#     record.append(ou_level)
#     random_record.append((random.randint(-50,50)*0.1))
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(record, 'k-', label='Ornstein Uhlenbeck noise')
# ax.grid(True)
# # ax.legend(loc='best')
# ax.set_xlabel(u"时间")
# ax.set_ylabel(u"噪音大小")
#
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1, 1, 1)
# ax1.plot(random_record, 'k-', label='random noise')
# ax1.set_xlabel(u"时间")
# ax1.set_ylabel(u"噪音大小")
# ax1.grid(True)
# # ax1.legend(loc='best')
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.show()

# total, max, min = 0, -1, 1
# for _ in range(100000):
#     ou_level = noise.ornstein_uhlenbeck_level(ou_level)
#     if ou_level > max:
#         max =ou_level
#     if ou_level < min:
#         min = ou_level
#     total += ou_level
# print max,min,total,total/100.0


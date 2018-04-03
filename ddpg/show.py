# -- coding: utf-8 --
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


class Drawer(object):

    def __init__(self, path, lineColor=['r-', 'b-', 'g-', 'y-']):
        self.lineColor = lineColor
        self.path = path

    def showBeta(self, idlist):
        beta_value = []
        if not isinstance(idlist, list):
            idlist = [idlist]

        for i in idlist:
            with open(self.path+'beta'+str(i)+'.csv', 'rb') as f:
                read = csv.reader(f)
                temp = []
                for line in read:
                    temp.append(line[0])
                beta_value.append(temp)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i, speed in enumerate(beta_value):
            mylabel = 'epoch '+str(idlist[i])
            ax.plot(speed, self.lineColor[i], label=mylabel, lw=1)
        ax.grid(True)
        ax.set_xlabel("time(s)")
        ax.set_ylabel("beta(m)")
        ax.legend(loc='best')
        plt.show()

    def showSpeed(self, idlist):
        speed_value = []
        if not isinstance(idlist, list):
            idlist = [idlist]
            idlist = [idlist]

        for i in idlist:
            with open(self.path+'speed'+str(i)+'.csv', 'rb') as f:
                read = csv.reader(f)
                temp = []
                for line in read:
                    temp.append(line[0])
                speed_value.append(temp)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i, speed in enumerate(speed_value):
            mylabel = 'epoch '+str(idlist[i])
            ax.plot(speed, self.lineColor[i], label=mylabel, lw=1)
        ax.grid(True)
        ax.set_xlabel("time(s)")
        ax.set_ylabel("speed(m)")
        ax.legend(loc='best')
        ax.set_aspect(1)
        plt.show()

    def showError(self, idlist):
        error_value = []
        if not isinstance(idlist, list):
            idlist = [idlist]

        for i in idlist:
            with open(self.path+'error'+str(i)+'.csv', 'rb') as f:
                read = csv.reader(f)
                temp = []
                for line in read:
                    temp.append(line[0])
                error_value.append(temp)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i, error in enumerate(error_value):
            mylabel = 'epoch '+str(idlist[i])
            ax.plot(error, self.lineColor[i], label=mylabel, lw=1)
        ax.grid(True)
        ax.set_xlabel("time(s)")
        ax.set_ylabel("error(m)")
        ax.legend(loc='best')
        plt.show()

    def showOrientation(self, idlist):
        action_value = []

        if not isinstance(idlist, list):
            idlist = [idlist]

        for i in idlist:
            with open(self.path+'action'+str(i)+'.csv', 'rb') as f:
                read = csv.reader(f)
                temp = []
                for line in read:
                    temp.append(line[1])
                action_value.append(temp)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i, value in enumerate(action_value):
            mylabel = 'epoch '+str(idlist[i])
            ax.plot(value, self.lineColor[i], label=mylabel, lw=1)
        ax.grid(True)
        ax.set_xlabel("time(s)")
        ax.set_ylabel("orientation torques(N)")
        ax.legend(loc='best')
        plt.show()

    def showRotation(self,idlist):
        action_value = []

        if not isinstance(idlist, list):
            idlist = [idlist]

        for i in idlist:
            with open(self.path+'action'+str(i)+'.csv', 'rb') as f:
                read = csv.reader(f)
                temp = []
                for line in read:
                    temp.append(line[0])
                action_value.append(temp)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i, value in enumerate(action_value):
            mylabel = 'epoch '+str(idlist[i])
            ax.plot(value, self.lineColor[i], label=mylabel, lw=1)
        ax.grid(True)
        ax.set_xlabel("time(s)")
        ax.set_ylabel("rotation torques(N)")
        ax.legend(loc='best')
        plt.show()

    def drawArc(self,p,r,s,e):
        x,y = p
        curr, step, res_x, res_y = s, 1, [], []

        while curr <= e:
            res_x.append(x + r * np.cos(curr))
            res_y.append(y + r * np.sin(curr))
            curr += step
        return res_x, res_y


    def showPath(self, idlist):
        path_value = []

        if not isinstance(idlist, list):
            idlist = [idlist]

        for i in idlist:
            with open(self.path+'wheelPath'+str(i)+'.csv','rb') as f:
                read = csv.reader(f)
                wheelx, wheely = [], []
                for line in read:
                    wheelx.append(line[0])
                    wheely.append(line[1])
                path_value.append([wheelx, wheely])

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # arc1 = list(self.drawArc((4,4),6,0, 0.5*np.pi))
        # arc2 = list(self.drawArc((0,14),4,np.pi, np.pi*1.5))
        # ax.plot(arc1[0],arc1[1], '-', lw=2)
        # ax.plot(arc2[0],arc2[1], '-', lw=2)
        # line1 = [(10, 0), (10, 4)]
        # line2 = [(4, 10), (0, 10)]
        # (line1_xs, line1_ys) = zip(*line1)
        # (line2_xs, line2_ys) = zip(*line2)
        # ax.add_line(Line2D(line1_xs, line1_ys, linewidth=2))
        # ax.add_line(Line2D(line2_xs, line2_ys, linewidth=2))

        for i, error in enumerate(path_value):
            mylabel = 'epoch '+str(idlist[i])
            ax.plot(error[0], error[1], self.lineColor[i], label=mylabel, lw=1)
        ax.set_aspect(1)
        # cir1 = Circle(xy=(0.0, 0.0), radius=5, alpha=0.4)
        # ax.add_patch(cir1)
        ax.set_xlabel("X(m)")
        ax.set_ylabel("Y(m)")
        ax.legend(loc='best')
        plt.show()

    def showLoss(self, idList):
        import csv
        loss, mylabel = [], ["with AE", "without AE", "3 node without AE"]
        for i in idList:
            with open(self.path+'loss'+str(i)+'.csv','rb') as f:
                reader = csv.reader(f)
                reader.next()
                v = []
                for row in reader:
                    v.append(float(row[2]))
                loss.append(v)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i, value in enumerate(loss):
            # mylabel = 'reward'+str(i)
            ax.plot(value, self.lineColor[i], label=mylabel[i], lw=1)
        # ax.grid(True)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        ax.set_xlabel(u"训练步数")
        ax.set_ylabel(u"残差")
        ax.legend(loc='best')
        # plt.xticks(range(0, 70, 10), range(0, 700, 100))
        plt.show()

    def showReward(self, idList):
        import csv
        reward, mylabel = [], ["with AE", "without AE", "3 node without AE"]
        for i in idList:
            with open(self.path+'Reward'+str(i)+'.csv','rb') as f:
                reader = csv.reader(f)
                reader.next()
                r = []
                for row in reader:
                    r.append(float(row[2]))
                reward.append(r)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i, value in enumerate(reward):
            # mylabel = 'reward'+str(i)
            ax.plot(value, self.lineColor[i], label=mylabel[i], lw=1)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # ax.grid(True)
        ax.set_xlabel(u"训练步数")
        ax.set_ylabel(u"奖励")
        ax.legend(loc='best')
        # plt.xticks(range(0, 70, 10), range(0, 700, 100))
        # plt.xticks(np.linspace(0,600,10))
        plt.show()







if __name__=="__main__":
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/final_csv/', [508, 530, 554]
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/true_csv/', [508, 530, 554]
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/', [0]
    path, modelNo = 'E:\experimentCode\deep-rl\ddpg//', [0, 1]
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/', [490] #, 233, 373]
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/change_csv/', [508, 529, 550]

    draw = Drawer(path)
    draw.showLoss(modelNo)
    draw.showReward(modelNo)
    # draw.showSpeed(modelNo)
    # draw.showError(modelNo)
    # draw.showPath(modelNo)
    # draw.showBeta(modelNo)
    # draw.showRotation(modelNo)

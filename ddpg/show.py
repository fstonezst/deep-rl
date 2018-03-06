# -- coding: utf-8 --
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


class Drawer(object):

    def __init__(self, path, lineColor=['-*', 'r-', 'g-', 'y-']):
        self.lineColor = lineColor
        self.path = path

    def showBeta(self, idlist):
        beta_value = []
        if not isinstance(idlist, list):
            idlist = [idlist]
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
        plt.xlim((0, 300))
        plt.xticks(range(0, 330, 30),[str(i) for i in range(0, 22, 2)])
        plt.yticks([(i/180.0) * np.pi for i in range(50,150,10)],[str(i) for i in range(50, 150, 10)])
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
            # mylabel = 'epoch '+str(idlist[i])
            mylabel = 'DRL-PF'
            ax.plot(error, self.lineColor[i], label=mylabel, lw=1)
            # ax.plot(error, self.lineColor[i], lw=1)
        ax.grid(True)
        ax.set_xlabel("Times(s)")
        ax.set_ylabel("Error(m)")
        # ax.legend(loc='best')
        plt.ylim((-0.08, 0.12))
        plt.xlim((0, 300))
        plt.xticks(range(0, 330, 30),[str(i) for i in range(0, 22, 2)])
        plt.yticks([i * 0.01 for i in range(-8, 14, 2)])
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
        plt.xlim((0, 300))
        plt.xticks(range(0, 330, 30),[str(i) for i in range(0, 22, 2)])
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
        plt.xlim((0, 300))
        plt.xticks(range(0, 330, 30),[str(i) for i in range(0, 22, 2)])
        plt.show()

    def drawArc(self,p,r,s,e):
        x,y = p
        curr, step, res_x, res_y = s, 0.017, [], []

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
            # with open(self.path+'movePath'+str(i)+'.csv','rb') as f:
                read = csv.reader(f)
                wheelx, wheely = [], []
                for line in read:
                    wheelx.append(line[0])
                    wheely.append(line[1])
                path_value.append([wheelx, wheely])

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        y = np.linspace(0, 4, 100)
        x = [10] * len(y)
        ax.plot(x,y,'k-', lw=1)
        x = np.linspace(4,0,100)
        y = [10] * len(x)
        ax.plot(x,y, 'k-', lw=1)
        y = np.linspace(14, 26, 100)
        x = [-4] * len(y)
        ax.plot(x,y, 'k-', lw=2)
        arc1 = list(self.drawArc((4,4),6,0, 0.5*np.pi))
        arc2 = list(self.drawArc((0,14),4,np.pi, np.pi*1.5))
        ax.plot(arc1[0],arc1[1], 'k-', lw=1)
        ax.plot(arc2[0],arc2[1], 'k-', lw=1)

        for i, error in enumerate(path_value):
            mylabel = 'epoch '+str(idlist[i])
            ax.plot(error[0], error[1], self.lineColor[i], label=mylabel, lw=1)
        ax.set_aspect(1)
        # cir1 = Circle(xy=(0.0, 0.0), radius=5, alpha=0.4)
        # ax.add_patch(cir1)
        ax.set_xlabel("X(m)")
        ax.set_ylabel("Y(m)")
        ax.legend(loc='best')
        plt.yticks(range(0, 22, 2))
        plt.ylim((0,20))
        plt.show()

    def showLoss(self, fileName='loss.csv'):
        import csv
        loss = []
        with open(self.path + fileName,'r') as f:
            reader = csv.reader(f)
            reader.next()
            for row in reader:
                loss.append(float(row[2]))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(loss, 'r-', label='loss', lw=1)
        ax.grid(True)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        # ax.legend(loc='best')
        plt.show()

    def showReward(self, fileName='Reward.csv'):
        import csv
        reward = []
        with open(self.path + fileName,'r') as f:
            reader = csv.reader(f)
            reader.next()
            for row in reader:
                reward.append(float(row[2]))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(reward, 'r-', label='reward', lw=1)
        ax.grid(True)
        ax.set_xlabel("step")
        ax.set_ylabel("reward")
        # ax.legend(loc='best')
        plt.show()







if __name__=="__main__":
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/final_csv/', [508, 530, 554]
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/true_csv/', [508, 530, 554]
    path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/', [436]
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/', [490] #, 233, 373]
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/change_csv/', [508, 529, 550]

    draw = Drawer(path)
    # draw.showLoss()
    # draw.showReward()
    # draw.showSpeed(modelNo)
    # draw.showError(modelNo)
    # draw.showPath(modelNo)
    draw.showBeta(modelNo)
    # draw.showRotation(modelNo)

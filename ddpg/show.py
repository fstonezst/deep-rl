import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class Drawer(object):

    def __init__(self, path, lineColor=['k-.', 'b--', 'r-', 'y-']):
        self.lineColor = lineColor
        self.path = path

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

    def showSpeed(self, idlist):
        speed_value = []
        if not isinstance(idlist, list):
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
        for i, value in enumerate(speed_value):
            mylabel = 'epoch '+str(idlist[i])
            ax.plot(value, self.lineColor[i], label=mylabel, lw=1)
        ax.grid(True)
        ax.set_xlabel("time(s)")
        ax.set_ylabel("speed(w/s)")
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

    def showPath(self, idlist, aspect=True):
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
                path_value.append([wheelx[:200], wheely[:200]])

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i, error in enumerate(path_value):
            mylabel = 'epoch '+str(idlist[i])
            ax.plot(error[0], error[1], self.lineColor[i], label=mylabel, lw=1)
        if aspect:
            ax.set_aspect(1)
        cir1 = Circle(xy=(0.0, 0.0), radius=5, alpha=0.4)
        ax.add_patch(cir1)
        ax.set_xlabel("X(m)")
        ax.set_ylabel("Y(m)")
        ax.legend(loc='best')
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
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/results/gym_ddpg/true_csv/', [508, 530, 554]
    path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/results/gym_ddpg/linearLog/', [738, 809, 832]
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/', [0]
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/true_csv/', [508, 530, 554]
    # path, modelNo = '/home/peter/PycharmProjects/deep-rl/ddpg/change_csv/', [508, 529, 550]

    draw = Drawer(path)
    # draw.showLoss()
    # draw.showReward()
    # draw.showError(modelNo)
    draw.showPath(modelNo, aspect=False)
    # draw.showRotation(modelNo)
    # draw.showOrientation(modelNo)
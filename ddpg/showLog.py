import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# import pandas as pd

def show(i):
    moveStorex, moveStorey=[], []
    wheelx, wheely = [], []
    action_r, action_s =[], []
    speed_reward, error_reward, total_reward = [], [], []
    path=sys.argv[1]

    with open(path+'movePath'+str(i)+'.csv','rb') as f:
        read = csv.reader(f)
        for line in read:
            moveStorex.append(line[0])
            moveStorey.append(line[1])

    with open(path+'wheelPath'+str(i)+'.csv','rb') as f:
        read = csv.reader(f)
        for line in read:
            wheelx.append(line[0])
            wheely.append(line[1])

    with open(path+'action'+str(i)+'.csv','rb') as f:
        read = csv.reader(f)
        for line in read:
            action_r.append(line[0])
            action_s.append(line[1])

    with open(path+'reward'+str(i)+'.csv','rb') as f:
        read = csv.reader(f)
        for line in read:
            speed_reward.append(line[0])
            error_reward.append(line[1])
            total_reward.append(float(line[0]) + float(line[1]))


    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax1 = fig.add_subplot(2,2,2)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4)
    ax.plot(moveStorex,moveStorey,'b-o',label='move_path')
    ax.plot(wheelx, wheely,'r-*',label='wheel_path')
    ax2.plot(action_r,'y-o',label='dir')
    ax1.plot(action_s,'r-*',label='speed')
    ax3.plot(error_reward,'r-o', label='error_reward')
    ax3.plot(speed_reward,'b-*', label='speed_reward')
    ax3.plot(total_reward,'y-*', label='total_reward')
    cir1 = Circle(xy=(0.0, 0.0), radius=10, alpha=0.4)
    ax.add_patch(cir1)
    # ax.plot(cir1, 'y-')
    plt.show()


if __name__=="__main__":
    import sys
    # show(sys./home/peter/Documents/log/showLog.py:45argv[2])
    show(1700)
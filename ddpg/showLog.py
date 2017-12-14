import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def show(i):
    moveStorex, moveStorey=[], []
    wheelx, wheely = [], []
    action_r, action_s =[], []
    with open('movePath'+str(i),'rb') as f:
        read = csv.reader(f)
        for x,y in read:
            moveStorex.append(x)
            moveStorey.append(y)

    with open('wheelPath'+str(i),'rb') as f:
        read = csv.reader(f)
        for x,y in read:
            wheelx.append(x)
            wheely.append(y)

    with open('action'+str(i),'rb') as f:
        read = csv.reader(f)
        for x,y in read:
            action_r.append(x)
            action_s.append(y)

    fig = plt.figure()
    ax = fig.add_subplot(3,1,1)
    ax1 = fig.add_subplot(3,1,2)
    ax2 = fig.add_subplot(3,1,3)
    ax.plot(moveStorex,moveStorey,'b-o',label='move_path')
    ax.plot(wheelx, wheely,'r-*',label='wheel_path')
    ax2.plot(action_r,'y-o',label='dir')
    ax1.plot(action_s,'r-*',label='speed')
    cir1 = Circle(xy=(0.0, 0.0), radius=10, alpha=0.4)
    ax.add_patch(cir1)
    plt.show()


if __name__=="__main__":
    import sys
    show(sys.argv[0])
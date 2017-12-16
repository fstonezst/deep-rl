# -- coding: utf-8 --
import numpy as np



class AGV:
    MAX_SPEED = 2 * np.pi
    MAX_ANGLE, MIN_ANGLE = np.pi * (170.0 / 180.0), np.pi * (10.0 / 180.0)
    MAX_ORIENTATION, MAX_ROTATION = 0.5, 300
    count = 0

    def __init__(self, mess=500, w_mess=[10, 1, 1], h=0.6, rs=0.125, rf=0.05, I0=250, Ip1=10, Ir=[1, 0.05, 0.05],
                 l=[1.22, 0.268, 0.268]):
        self.wheelPos = [10, 0]
        self.uk = np.matrix([[0.0], [0.0]])
        self.fai = np.matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        self.D = np.matrix([0, 0, 1]).T
        self.L = np.matrix([[0, 0, 0], [0, 0, l[0]], [0, -1 / l[0], 0]])
        self.T = np.matrix([[0, 0, Ip1]])

        xita = np.pi
        x = self.wheelPos[0] - (np.cos(xita - np.pi / 2.0) * l[0])
        y = self.wheelPos[1] - (np.sin(xita - np.pi / 2.0) * l[0])
        posState = np.array([x, y, xita]).T

        B = np.pi / 2.0
        rollSpeed = np.array([0, 0, 0]).T
        self.q = np.reshape(np.concatenate((posState.T, np.array([B]), rollSpeed)), [7, 1])

        self.mess, self.w_mss, self.h = mess, np.array(w_mess), h
        self.I0, self.Ip1, self.Ir, self.l = I0, Ip1, np.array(Ir), np.array(l)
        self.rs, self.rf, self.Ib = rs, rf, self.Ip1

        m11 = m22 = self.mess + sum(self.w_mss)
        m12 = m21 = 0
        m13 = m31 = self.mess * self.h + self.w_mss[0] * self.l[0]
        m23 = m32 = -self.w_mss[1] * self.l[1] + self.w_mss[2] * self.l[2]
        m33 = self.mess * self.h * self.h + self.I0 + sum(self.w_mss * self.l * self.l)

        self.M = np.matrix([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
        self.V = np.array([0, 0, self.Ip1]).T
        self.If = np.matrix([[self.Ir[0], 0, 0], [0, self.Ir[1], 0], [0, 0, self.Ir[2]]])
        self.J2 = np.matrix([[self.rs, 0, 0], [0, self.rf, 0], [0, 0, self.rf]])
        self.P = np.matrix([1, 0, 0]).T

    def setMess(self, m):
        if m > 0 :
            self.mess = m

    def countHB(self):
        return np.dot(np.dot(self.getSumB().T, self.M),
                      np.dot(np.dot(np.dot(self.getEB().T, self.If), self.getEB()), self.getSumB()))

    def countTba(self):
        Tba4 = np.dot(np.dot(self.getSumB().T, self.getEB().T), self.P)
        return np.matrix([[1, 0], [0, Tba4]])

    def countMba(self):
        Mba1 = float(np.dot(self.V.T, self.getSumB()))
        Mba2 = float(np.dot(self.getSumB().T, self.V))
        return np.matrix([[Mba1, self.Ib], [self.countHB(), Mba2]])

    def getEB(self):
        return -np.dot(self.J2.I, self.getJ1B())

    def getJ1B(self):
        sinB = np.sin(float(self.q[3]))
        cosB = np.cos(float(self.q[3]))
        return np.matrix([[cosB, sinB, float(self.l[0]) * cosB],
                          [0, 1, float(self.l[1])], [0, -1, float(self.l[1])]])

    def getUk(self, ud):
        return np.dot(np.dot(self.countMba().I, self.countTba()), ud) - np.dot(self.countMba().I, self.countVba())

    def getR(self):
        xita = float(self.q[2])
        sinxita = np.sin(xita)
        cosxita = np.cos(xita)
        return np.matrix([[cosxita, sinxita, 0], [-sinxita, cosxita, 0], [0, 0, 1]])

    def getS(self):
        R = self.getR()
        sumB = self.getSumB()
        s11 = np.dot(R.T, sumB)
        s31 = np.dot(self.getEB(), sumB)
        s1 = np.concatenate((s11, np.matrix([[0]]), s31), axis=0)
        s2 = [[0]] * 3
        s2.append([1])
        s2.extend([[0]] * 3)
        s2 = np.matrix(s2)

        return np.concatenate((s1, s2), axis=1)

    def getSumB(self):
        B = float(self.q[3])
        return np.matrix([[0, (-self.rs * np.sin(B)), (-(self.rs * np.cos(B)) / float(self.l[0]))]]).T

    def setQ(self, q):
        self.q = q

    def setB(self, b):
        self.q[3] = b

    def control(self, dk):
        self.uk = self.uk + dk


        #constraint of angle and speed of steering wheel
        angle = float(self.q[3])
        if angle <= AGV.MIN_ANGLE:
            #right direction
            self.uk[1] = 0
            self.q[3] = AGV.MIN_ANGLE
        elif angle >= AGV.MAX_ANGLE:
            #left direction
            self.uk[1] = 0
            self.q[3] = AGV.MAX_ANGLE

        if self.uk[0] > AGV.MAX_SPEED:
            self.uk[0] = AGV.MAX_SPEED
        elif self.uk[0] < 0:
            self.uk[0] = 0


        #debug
        # AGV.count += 1
        # print AGV.count,':',self.uk[0],self.uk[1],self.q[3]



        s = self.getS()
        dq = np.dot(s, self.uk)

        self.q = self.q + dq

        # if float(self.q[2]) > 2 * np.pi:
        #     self.q[2] %= (2 * np.pi)
        # elif self.q[2] < -2 * np.pi:
        #     self.q[2] = -(abs(float(self.q[2]) % (2 * np.pi)))


        x, y, xita, l = float('%.8f' % self.q[0]), float('%.8f' % self.q[1]), float('%.8f' % self.q[2]), self.l[0]
        cosres = float('%.8f' % np.cos(xita - np.pi / 2.0))
        sinres = float('%.8f' % np.sin(xita - np.pi / 2.0))

        wx = x + cosres * l
        wy = y + sinres * l
        self.wheelPos[0] = wx
        self.wheelPos[1] = wy

    def getk1(self):
        uk = self.uk
        res1 = np.dot(self.getR().T, self.fai.T)
        res1 = np.dot(res1, self.getSumB())
        res1 = np.dot(res1, self.D.T)
        res1 = np.dot(res1, self.getR().T)
        res1 = np.dot(res1, self.getSumB())
        res1 = res1 * float(uk[0]) * float(uk[0])

        res2 = np.dot(self.getR().T, self.L)
        res2 = np.dot(res2, self.getSumB())
        res2 = res2 * float(uk[0]) * float(uk[1])

        return res1 + res2

    def getNb(self):
        B = float(self.q[3])
        sinb = np.sin(B)
        cosb = np.cos(B)
        return np.matrix([[-sinb, cosb, -float(self.l[0]) * sinb], [0, 0, 0], [0, 0, 0]])

    def getk2(self):
        uk = self.uk
        res1 = np.dot(self.J2.I, self.getNb())
        res2 = np.dot(self.getEB(), self.L)
        res2 = res1 - res2
        res = np.dot(res2, self.getSumB())
        return res * float(uk[0]) * float(uk[1])

    def countf1(self):
        return np.dot(np.dot(self.V.T, self.getR()), self.getk1())

    def countf2(self):
        n_ = float(self.uk[0])
        kexi_ = float(self.uk[1])
        step1 = np.dot(self.fai.T, self.M) + np.dot(self.M, self.fai)
        step1 = np.dot(self.getSumB().T, step1)
        step1 = np.dot(step1, self.getSumB())
        step1 = np.dot(step1, self.D.T)
        step1 = np.dot(step1, self.getR().T)
        step1 = np.dot(step1, self.getSumB())
        step1 = step1 * n_ * n_

        step2 = np.dot(self.getSumB().T, self.M)
        step2 = np.dot(step2, self.getR())
        step2 = np.dot(step2, self.getk1())

        step3 = np.dot(self.getSumB().T, self.fai.T)
        step3 = np.dot(step3, self.V)
        step3 = np.dot(step3, self.D.T)
        step3 = np.dot(step3, self.getR().T)
        step3 = np.dot(step3, self.getSumB())
        step3 = step3 * n_ * kexi_

        step4 = np.dot(self.getSumB().T, self.getR())
        step4 = np.dot(step4, self.D)
        temp1 = np.dot(self.getSumB().T, self.M) * n_
        temp2 = self.V.T * kexi_
        temp = temp1 + temp2
        step4 = np.dot(step4, temp)
        step4 = np.dot(step4, self.fai)
        step4 = np.dot(step4, self.getSumB()) * n_

        step5 = np.dot(self.getSumB().T, self.getEB())
        step5 = np.dot(step5, self.getk2())

        return step1 + step2 + step3 + step4 + step5

    def countVba(self):
        return np.matrix([[float(self.countf1()), float(self.countf2())]]).T

    def controlInput(self, input):
        a, b, c = self.countMba().I, self.countTba(), self.countVba()
        # step1 = np.dot(self.countMba().I, self.countTba())
        step1 = np.dot(a, b)
        step1 = np.dot(step1, input.T)
        # step2 = np.dot(self.countMba().I, self.countVba())
        step2 = np.dot(a, c)
        dk = (step1 - step2)
        self.control(dk)


# import matplotlib.pyplot as plt
# car = AGV()
# pathx, pathy = [], []
# centerx, centery = [], []
# B = []
# for i in range(100):
#     pathx.append(car.q[0])
#     pathy.append(car.q[1])
#     centerx.append(car.wheelPos[0])
#     centery.append(car.wheelPos[1])
#     # car.control(np.matrix([[1], [2]]))
#     # car.controlInput(np.matrix([5, 2000]))
#     if i < 50:
#         car.controlInput(np.matrix([0.05, 50]))
#     else:
#         car.controlInput(np.matrix([-0.05, 70]))
#     B.append(float(car.q[2]))
#     print car.uk[0],car.q[3]
#         # car.controlInput(np.matrix([0,0]))

# car = AGV(mess=1000)
# car = AGV(mess=1000,w_mess=[15,1.8,1.8],Ir=[1.3, 0.03, 0.03], Ip1=14)
# pathx1, pathy1 = [], []
# centerx1, centery1 = [], []
# B1 = []
# for i in range(100):
#     pathx1.append(car.q[0])
#     pathy1.append(car.q[1])
#     centerx1.append(car.wheelPos[0])
#     centery1.append(car.wheelPos[1])
#     # car.control(np.matrix([[1], [2]]))
#     # car.controlInput(np.matrix([5, 2000]))
#     if i < 50:
#         car.controlInput(np.matrix([0.1, 50]))
#     else:
#         car.controlInput(np.matrix([-0.1, -50]))
#     B1.append(float(car.q[2]))
#     print car.uk[0],car.q[3]
#         # car.controlInput(np.matrix([0,0]))
#
# fig = plt.figure()
# ax = fig.add_subplot(2,1,1)
# ax1 = fig.add_subplot(2,1,2)
#
# ax.plot(pathx,pathy,'r-o')
# ax.plot(centerx,centery,'b-*')
# ax1.plot(B,'b-*')
# ax.plot(pathx1,pathy1,'r-*')
# ax.plot(centerx1,centery1,'b-o')
# ax1.plot(B1,'y-*')
#
# plt.show()


import numpy as np

class AGV:
    def __init__(self, mess=500, w_mess=[10, 1, 1], h=0.6, rs=0.125, rf=0.05, I0=250, Ip1=10, Ir=[1, 0.05, 0.05],
                 l=[1.22, 0.268, 0.268]):

        self.fai = np.matrix([[0,1,0],[-1,0,0],[0,0,0]])
        self.D = np.matrix([0,0,1]).T
        self.L = np.matrix([[0,0,0],[0,0,l[0]],[0,-1/l[0],0]])

        posState = np.array([0, 0, np.pi/2.0]).T
        B = 0
        rollSpeed = np.array([0, 0, 0]).T
        self.q = np.reshape(np.concatenate((posState.T, np.array([B]), rollSpeed)),[7,1])

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
        self.If = np.matrix([[self.Ir[0],0,0],[0,self.Ir[1],0],[0,0,self.Ir[2]]])

        self.sumB = self.getSumB()
        # self.sumB = np.matrix([[0, (-self.rs * np.sin(self.q[1])), (-(self.rs * np.cos(self.q[1])) / self.l[0])]]).T

        self.J1B = self.getJ1B()

        self.J2 = np.matrix([[self.rs, 0, 0], [0, self.rf, 0], [0, 0, self.rf]])

        self.EB = self.getEB()

        self.HB = np.dot(np.dot(self.sumB.T,self.M),np.dot(np.dot(np.dot(self.EB.T,self.If),self.EB),self.sumB))
        self.P = np.matrix([1,0,0]).T


        Mba1 = float(np.dot(self.V.T,self.sumB))
        Mba2 = float(np.dot(self.sumB.T,self.V))
        self.Mba = np.matrix([[Mba1,self.Ib],[self.HB, Mba2]])

        Tba4=np.dot(np.dot(self.sumB.T,self.EB.T),self.P)
        self.Tba = np.matrix([[1,0],[0,Tba4]])


        # print self.Mba
        # print self.Tba
    def getEB(self):
        return -np.dot(self.J2.I, self.getJ1B())

    def getJ1B(self):
        return np.matrix([[np.cos(float(self.q[1])), np.sin(float(self.q[1])), float(self.l[0]) * np.cos(float(self.q[1]))],
                             [0, 1, float(self.l[1])], [0, -1, float(self.l[1])]])


    def getUk(self,ud):
        return np.dot(np.dot(self.Mba.I,self.Tba),ud) - np.dot(self.Mba.I,self.V)

    def getR(self):
        xita = float(self.q[2])
        sinxita = np.sin(xita)
        cosxita = np.cos(xita)
        return np.matrix([[cosxita,sinxita,0],[-sinxita,cosxita,0],[0,0,1]])

    def getS(self):
        s1= np.concatenate((np.dot(self.getR().T,self.getSumB()),np.matrix([[0]]),np.dot(self.EB,self.sumB)),axis=0)
        s2 = [[0]]*3
        s2.append([1])
        s2.extend([[0]]*3)
        s2 = np.array(s2)

        return np.concatenate((s1,s2),axis=1)

    def getSumB(self):
        B = self.q[3]
        return np.matrix([[0, (-self.rs * np.sin(B)), (-(self.rs * np.cos(B)) / self.l[0])]]).T

    def setQ(self,q):
        self.q = q
    def setB(self,b):
        self.q[3] = b

    def control(self,uk):
        self.setB(uk[1])
        s = self.getS()
        dq = np.dot(s,uk)
        self.q = self.q + dq

    def getk1(self,uk):

        res1 = np.dot(self.getR().T,self.fai.T)
        res1 = np.dot(res1,self.getSumB())
        res1 = np.dot(res1,self.D.T)
        res1 = np.dot(res1,self.getR().T)
        res1 = np.dot(res1,self.getSumB())
        res1 = res1 * float(uk[0]) * float(uk[0])

        res2 = np.dot(self.getR().T,self.L)
        res2 = np.dot(res2,self.getSumB())
        res2 = res2 * float(uk[0]) * float(uk[1])

        return res1 + res2

    def getNb(self):
        B = float(self.q[3])
        sinb = np.sin(B)
        cosb = np.cos(B)
        return np.matrix([[-sinb,cosb,-float(self.l[0])*sinb],[0,0,0],[0,0,0]])

    def getk2(self,uk):
        res1 = np.dot(self.J2.T,self.getNb())
        res2 = np.dot(self.EB,self.L)
        res2 = res1 - res2
        res = np.dot(res2,self.getSumB())
        return res * float(uk[0]) * float(uk[1])




a = AGV()

print a.getk1(np.matrix([[100],[0]]))
# print "======control====="
# a.control(np.matrix([[100],[0]]))
# print a.q

# a.setB(0)
# print np.dot(a.getS(),np.matrix([[100],[0]]))


import numpy as np
import matplotlib.pyplot as plt

class TWO_LINK_MANIPULATOR:
    def __init__(self, p1, p2, p3, qr, q0, D, Kp, ks = 1):
        self.q = q0 
        self.q_dot = np.zeros(np.shape(self.q),dtype = 'float64' )
        self.q_dot_dot = np.zeros(np.shape(self.q), dtype = 'float64')
        self.qr = qr
        self.e = self.q - self.qr #defining error term
        
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.m1 = 2.5
        self.m2 = 1.5
        self.l = 0.4
        self.g = 9.81
        self.dt = 0.01

        self.D = D 
        self.Kp =  Kp
        self.ks = ks # control law gain
        # ks is set to one by default
        
        # Initial variable updation
        self.q1 = self.q[0]
        self.q2 = self.q[1]
        self.q1_dot = self.q_dot[0]
        self.q2_dot = self.q_dot[1]

        self.M = np.array([[self.p1 + (2 * self.p3 * np.cos(self.q2)), self.p2 + (self.p3 * np.cos(self.q2))],[(self.p3 * np.cos(self.q2)), self.p2]])
        self.C = self.p3 * np.sin(self.q2) * np.array([[-self.q2_dot, self.q1_dot + self.q2_dot],[self.q1_dot, 0]])
        self.g_eff = self.g * np.array([(self.l * (self.m1 + self.m2) * np.cos(self.q1)) + (self.l * self.m2 * np.cos(self.q1 + self.q2)), self.l * self.m2 * np.cos(self.q2)])      

    def dynamic_state_updation(self):
        self.q1 = self.q[0]
        self.q2 = self.q[1]
        self.q1_dot = self.q_dot[0]
        self.q2_dot = self.q_dot[1]
        
        self.M = np.array([[self.p1 + (2 * self.p3 * np.cos(self.q2)), self.p2 + (self.p3 * np.cos(self.q2))],[(self.p3 * np.cos(self.q2)), self.p2]])
        self.C = self.p3 * np.sin(self.q2) * np.array([[-self.q2_dot, self.q1_dot + self.q2_dot],[self.q1_dot, 0]])
        self.g_eff = self.g * np.array([(self.l * (self.m1 + self.m2) * np.cos(self.q1)) + (self.l * self.m2 * np.cos(self.q1 + self.q2)), self.l * self.m2 * np.cos(self.q2)])      

        self.e = self.q - self.qr
        self.q_dot_dot = np.matmul(np.linalg.inv(self.M),(-self.ks * np.tanh(self.q_dot) - np.matmul(self.C,self.q_dot) - np.matmul(self.D, self.q_dot) - np.matmul(self.Kp ,self.e)))
        self.q_dot += self.q_dot_dot * self.dt
        self.q += self.q_dot * self.dt
    
    def iterate(self, t_f):
        self.t = np.linspace(0,t_f,int(t_f/self.dt))
        self.u_array = []
        self.q_array = []
        self.qr_array = []
        self.q1_array = []
        self.q1_r_array = []
        self.q2_array = []
        self.q2_r_array = []
        self.u1_array = []
        self.u2_array = []
        for _ in range(len(self.t)):
            u = self.g_eff - np.matmul(self.Kp , self.e) - self.ks * np.tanh(self.q_dot)
            self.u_array.append(np.linalg.norm(u))
            self.q_array.append(np.linalg.norm(self.q))
            self.qr_array.append(np.linalg.norm(self.qr))
            self.q1_array.append(self.q[0])
            self.q2_array.append(self.q[1])
            self.u1_array.append(u[0])
            self.u2_array.append(u[1])
            self.q1_r_array.append(self.qr[0])
            self.q2_r_array.append(self.qr[1])

            self.dynamic_state_updation()
        

    def plot(self):
        plt.subplots(1)
        plt.plot(self.t, self.u_array, 'red')
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Magnitude of u v/s Time")
    
        plt.subplots(1)
        plt.plot(self.t, self.q_array, 'blue')
        plt.plot(self.t, self.qr_array, 'k--', label = r"$q_r$", linewidth = 2)
        plt.xlabel("t")
        plt.ylabel(r"$q$")
        plt.title("Magnitude of q v/s Rime")
    
        plt.subplots(1)
        plt.plot(self.t, self.u1_array, 'red', label = r"$u_1$")
        plt.plot(self.t, self.u2_array, 'black', label = r"$u_2$")
        plt.xlabel("t")
        plt.title("Variation of Control Input components v/s Time")
    
        plt.subplots(1)
        plt.plot(self.t, self.q1_array, 'red', label = r"$q_1$")
        plt.plot(self.t, self.q1_r_array, 'r--', label = r"$q_{1r}$")
        plt.plot(self.t, self.q2_array, 'black', label = r"$q_2$")
        plt.plot(self.t, self.q2_r_array, 'k--', label = r"$q_{2r}$")
        plt.xlabel("t")
        plt.title("Variation of State components v/s Time")
        plt.legend()

        plt.show()
    
if __name__ == '__main__':
    # Roll No. : 20D110021
    X = 1.
    p1 = 3.31 + (X/30)
    p2 = 0.116 + (X/500)
    p3 = 0.16 + (X/400)

    qr = np.array([0.4, 0], dtype = 'float64')
    q0 = np.array([-1, 3], dtype = 'float64')


    # As of here, I have take them as Identity Matrices, such that D is a positive semi-definite matrix and K_p is a symmetric positive definite
    D = np.array([[1, 0 ],[0, 1]], dtype = 'float64')
    Kp = np.array([[1, 0 ],[0, 1]],dtype = 'float64')
    
    manipulator = TWO_LINK_MANIPULATOR(p1, p2, p3, qr, q0, D, Kp)
    manipulator.iterate(20)
    print("Initial input:", q0)
    print("Kp:", Kp)
    manipulator.plot()

    # also, you may pass ks (control law gain) which is by defalt set to 1 to observe the change
    
    #########################################################################

    # uncomment the below lines to see convergence on different initial values

    # q1 = np.array([1, 0], dtype = 'float64')
    # q2 = np.array([3,6], dtype = 'float64')
    # q3 = np.array([-2, 7], dtype = 'float64')

    # manipulator1 = TWO_LINK_MANIPULATOR(p1, p2, p3, qr, q0, D, Kp)
    # manipulator2 = TWO_LINK_MANIPULATOR(p1, p2, p3, qr, q1, D, Kp)
    # manipulator3 = TWO_LINK_MANIPULATOR(p1, p2, p3, qr, q2, D, Kp)
    # manipulator4 = TWO_LINK_MANIPULATOR(p1, p2, p3, qr, q3, D, Kp)

    # manipulator1.iterate(20)
    # manipulator2.iterate(20)
    # manipulator3.iterate(20)
    # manipulator4.iterate(20)

    # plt.subplots(1)
    # plt.plot(manipulator1.t, manipulator1.q_array, 'blue', label = "q_0 = [1, 2]")
    # plt.plot(manipulator2.t, manipulator2.q_array, 'green',label = "q_0 = [0, 1]")
    # plt.plot(manipulator3.t, manipulator3.q_array, 'orange',label = "q_0 = [3, 6]")
    # plt.plot(manipulator4.t, manipulator4.q_array, 'red',label = "q_0 = [-1, 5]")
    # plt.plot(manipulator1.t, manipulator1.qr_array, 'k--', label = r"$q_r$", linewidth = 2)
    # plt.xlabel("t")
    # plt.ylabel(r"$q$")
    # plt.title("Magnitude of q v/s Time")
    # plt.legend()
    # plt.show()

    #####################################################################################
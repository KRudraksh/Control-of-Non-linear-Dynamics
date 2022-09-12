import numpy as np
import matplotlib.pyplot as plt

class TWO_LINK_MANIPULATOR_SCALAR:
    def __init__(self, p1, p3, q0, qr):
        self.q = q0    
        self.qr = qr
        self.e = self.q - self.qr
        self.q_dot = 0.
        self.q_dot_dot = 0.
        self.xi = 0.
        self.xi_dot = 0
        self.p1 = p1
        self.p3 = p3
        self.dt = 0.1

        # Initial variable updation
        self.m = self.p1 + (2 * self.p3 * np.cos(self.q))
        self.c = -self.p3 * np.sin(self.q) * self.q_dot
        self.m_dot = -2 * self.p3 * np.sin(self.q) * self.q_dot
        self.c_dot = -(self.p3 * np.cos(self.q) * (self.q_dot**2)) - (self.p3 * np.sin(self.q) * self.q_dot_dot)

    def dynamic_state_updation(self, tau):
        # dynamics
        self.q_dot_dot = (self.xi - (self.c * self.q_dot)) / self.m
        self.xi_dot = tau        

        # update states
        self.q_dot += self.q_dot_dot * self.dt
        self.q += self.q_dot * self.dt
        self.xi += self.xi_dot * self.dt
        self.m = self.p1 + (2 * self.p3 * np.cos(self.q))
        self.c = -self.p3 * np.sin(self.q) * self.q_dot
        self.m_dot = -2 * self.p3 * np.sin(self.q) * self.q_dot
        self.c_dot = -(self.p3 * np.cos(self.q) * (self.q_dot**2)) - (self.p3 * np.sin(self.q) * self.q_dot_dot)

    def compute_control_law(self): # taking the control lyapunov function to be V(q,q_dot) = e^2/2 + q_dot^2/2
        k_0_dot = (self.c_dot * self.q_dot + self.c * self.q_dot_dot) + self.m_dot * self.qr - (self.m_dot*self.q + self.m*self.q_dot) - (self.m_dot*self.q_dot + self.m*self.q_dot_dot)
        self.tau = k_0_dot - (2*self.q_dot/self.m) - (self.xi/(self.m**2)) + (self.c*self.q_dot/(self.m**2)) + ((self.qr - self.q)/self.m)

    def iterate(self, tf) :
        self.t = np.linspace(0,tf,int(tf/self.dt))
        self.compute_control_law()
        self.tau_array = []
        self.q_array = []
        self.qr_array = []
        for _ in range (len(self.t)):
            self.tau_array.append(self.tau)
            self.q_array.append(self.q)
            self.qr_array.append(self.qr)
            self.dynamic_state_updation(self.tau)
            self.compute_control_law()
        
    def plot(self):
        plt.subplots(1)
        plt.plot(self.t, self.tau_array, 'red')
        plt.xlabel("t")
        plt.ylabel(r"$\tau$")
        plt.title(r"Control Torque $\tau$ vs Time")
    
        plt.subplots(1)
        plt.plot(self.t, self.q_array, 'blue')
        plt.plot(self.t, self.qr_array, 'k--', label = r"$q_r$", linewidth = 2)
        plt.xlabel("t")
        plt.ylabel(r"$q$")
        plt.title("Angle q vs Time")
    
        plt.show()

if __name__ == '__main__' :
    # Roll No. : 20D110021
    X = 1.

    p_1 = 3.31 + (X/30)
    p_3 = 0.16 + (X/400)

    q_r = 0.3
    q_0 = 0.5

    manipulator = TWO_LINK_MANIPULATOR_SCALAR(p_1,p_3,q_0,q_r)
    manipulator.iterate(150)
    manipulator.plot()

    ##################################################################################

    # uncomment the below lines to see convergence on different initial values

    # q_1 = -3
    # q_2 = -8
    # q_3 = 6
    # manipulator1 = TWO_LINK_MANIPULATOR_SCALAR(p_1,p_3,q_0,q_r)
    # manipulator2 = TWO_LINK_MANIPULATOR_SCALAR(p_1,p_3,q_1,q_r)
    # manipulator3 = TWO_LINK_MANIPULATOR_SCALAR(p_1,p_3,q_2,q_r)
    # manipulator4 = TWO_LINK_MANIPULATOR_SCALAR(p_1,p_3,q_3,q_r)
    # manipulator1.iterate(150)
    # manipulator2.iterate(150)
    # manipulator3.iterate(150)
    # manipulator4.iterate(150)

    # plt.plot(manipulator1.t, manipulator1.q_array, 'blue', label = "q_0 = 1")
    # plt.plot(manipulator2.t, manipulator2.q_array, 'green',label = "q_0 = 3.5")
    # plt.plot(manipulator3.t, manipulator3.q_array, 'orange',label = "q_0 = -2.9")
    # plt.plot(manipulator4.t, manipulator4.q_array, 'red',label = "q_0 = 7")
    # plt.plot(manipulator1.t, manipulator1.qr_array, 'k--', label = r"$q_r$", linewidth = 2)   
    # plt.xlabel("t")
    # plt.ylabel(r"$q$")
    # plt.title("Angle q vs time t")
    # plt.legend()
    # plt.show() 

    #############################################################################
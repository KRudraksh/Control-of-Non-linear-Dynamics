import numpy as np
import matplotlib.pyplot as plt

class RIGID_BODY_ROTATIONAL_DYMANICS:
    def __init__(self, alpha_1, rho_0, omega_0):
        self.omega = omega_0
        self.rho = rho_0
        self.J = np.array([[20 + alpha_1, 1.2, 0.9],[1.2, 17 + alpha_1, 1.4],[0.9, 1.4, 15 + alpha_1]])
        self.rho_cross = np.array([[0, - self.rho[2], self.rho[1]],[self.rho[2], 0, -self.rho[0]],[-self.rho[1], self.rho[0], 0]])
        self.I = np.identity(3)
        self.dt = 0.1
        self.k_s = 1.2 # storage function gain
        self.k_c = 14 # control law gain
        
        self.compute_jacobian()
    
    # now compute the jacobian of the storage function to compute the control law
    def compute_jacobian(self):
        self.dW_drho = ((2 * self.k_s )/ (1 + np.matmul(np.transpose(self.rho), self.rho))) * np.transpose(self.rho)

    def dynamics_state_updation(self, u) :
        # dynamics
        self.rho_dot = np.matmul((self.I + self.rho_cross + np.matmul(self.rho, np.transpose(self.rho))), self.omega)
        self.omega_dot = np.matmul(np.linalg.inv(self.J),-(np.cross(self.omega , np.matmul(self.J,self.omega))) + u)
        self.y = self.omega
        self.compute_jacobian()
        
        # update states
        self.omega = (self.omega_dot * self.dt) + self.omega
        self.rho = (self.rho_dot * self.dt) + self.rho
        self.rho_cross = np.array([[0, - self.rho[2], self.rho[1]],[self.rho[2], 0, -self.rho[0]],[-self.rho[1], self.rho[0], 0]])
    
    def control_law(self):
        self.nu = -self.k_c * np.tanh(self.omega)
        self.u = self.nu - np.transpose(np.matmul(self.dW_drho,(self.I + self.rho_cross + np.matmul(self.rho, np.transpose(self.rho))))) 
        

    def iterate(self, t_f):
        self.t = np.linspace(0,t_f,int(t_f/self.dt))
        self.control_law()

        self.u_array = []
        self.rho_array = []
        self.omega_array = []

        for _ in range (len(self.t)):
            self.u_array.append(np.linalg.norm(self.u))
            self.rho_array.append(np.linalg.norm(self.rho))
            self.omega_array.append(np.linalg.norm(self.omega))

            self.dynamics_state_updation(self.u)
            self.control_law()

    def plot(self):
        # plot of control with respect to time
        plt.subplots(1)
        plt.plot(self.t, self.u_array, 'red')
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Variation of Control v/s time")

        # plot of omega with respect to time
        plt.subplots(1)
        plt.plot(self.t, self.omega_array, 'blue')
        plt.xlabel("t")
        plt.ylabel(r"$\omega(t)$")
        plt.title("Variation of " + r"$\omega(t)$" + " v/s time")
        
        # plot of rho with respect to time
        plt.subplots(1)
        plt.plot(self.t, self.rho_array, 'black')
        plt.xlabel("t")
        plt.ylabel(r"$\rho(t)$")
        plt.title("Variation of " + r"$\rho(t)$" + " v/s time")

        plt.show()

if __name__ == '__main__':
    # Roll_No := 20D110021
    alpha_1 = 0.21
    rho_0 = np.transpose(np.array([-0.02, 0, 0.045]))
    omega_0 = np.transpose(np.array([0.004, -0.007, 0.017]))
    rot_dyn = RIGID_BODY_ROTATIONAL_DYMANICS(alpha_1, rho_0, omega_0)
    rot_dyn.iterate(20)
    print("Storage Function gain:", rot_dyn.k_s)
    print("Control Law Gain:", rot_dyn.k_c)
    rot_dyn.plot()
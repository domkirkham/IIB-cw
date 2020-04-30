import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import random

g_na = 120
g_k = 36
g_l = 0.3

e_na = 50
e_k = -77
e_l = -54.4

class Neuron:
    """Neuron class"""
    def __init__(self):
        """Initialise neuron activity as bool"""
        self.v = -65
        self.i_ext = 0
        self.dt = 0.001
        
    def alpha_m(self, step):
        return (2.5-0.1*(self.v_hist[step]+65)) / (np.exp(2.5-0.1*(self.v_hist[step]+65)) -1)
    
    def beta_m(self, step):
        return 4*np.exp(-(self.v_hist[step]+65)/18)
    
    def alpha_h(self, step):
        return 0.07*np.exp(-(self.v_hist[step]+65)/20)
    
    def beta_h(self, step):
        return 1/(np.exp(3.0-0.1*(self.v_hist[step]+65))+1)
    
    def alpha_n(self, step):
        return (0.1-0.01*(self.v_hist[step]+65)) / (np.exp(1-0.1*(self.v_hist[step]+65)) -1)
    
    def beta_n(self, step):
        return 0.125*np.exp(-(self.v_hist[step]+65)/80)
    
    def grad_v(self, step, var=False):
        if var:
            return (-g_na * self.m_hist[step]**3 * self.h_hist[step] * (self.v_hist[step]-e_na)) -(g_k * self.n_hist[step]**4 *(self.v_hist[step]-e_k)) -(g_l *(self.v_hist[step]-e_l)) + self.i_ext[step]
        else:
            return (-g_na * self.m_hist[step]**3 * self.h_hist[step] * (self.v_hist[step]-e_na)) -(g_k * self.n_hist[step]**4 *(self.v_hist[step]-e_k)) -(g_l *(self.v_hist[step]-e_l)) + self.i_ext
    
    def grad_m(self, step):
        return self.alpha_m(step) * (1-self.m_hist[step]) - self.beta_m(step)*self.m_hist[step]
    
    def grad_h(self, step):
        return self.alpha_h(step) * (1-self.h_hist[step]) - self.beta_h(step)*self.h_hist[step]
    
    def grad_n(self, step):
        return self.alpha_n(step) * (1-self.n_hist[step]) - self.beta_n(step)*self.n_hist[step]
    
    def generate_input(self, T=np.arange(10, 20, 1), p=5, I=2.3):
        total_length = int(sum(T) // self.dt)
        p_length = int(p // self.dt)
        input_current = np.zeros((total_length))
        for i, period in enumerate(T):
            start_loc = int(sum(T[:i])//self.dt) - p_length
            end_loc = int(sum(T[:i])//self.dt)
            input_current[start_loc:end_loc] = I
        return input_current
    
    def simulate_1(self, i_ext=np.arange(-5, 6, 1), plot=True):
        
        frequencies = np.zeros((len(i_ext)))
        
        for num, i in enumerate(i_ext):
            print(f"I_ext = {i}")
            self.v = -65
            self.m = 0
            self.h = 0
            self.n = 0
            self.i_ext = i
            self.num_steps = int(200//self.dt)
            self.v_hist = np.zeros((self.num_steps))
            self.m_hist = np.zeros((self.num_steps))
            self.n_hist = np.zeros((self.num_steps))
            self.h_hist = np.zeros((self.num_steps))
            
            # # Initialise resting conditions for gating variables
            self.v_hist[0] = self.v
            self.m = self.alpha_m(0)/ (self.alpha_m(0) + self.beta_m(0))
            self.n = self.alpha_n(0)/ (self.alpha_n(0) + self.beta_n(0))
            self.h = self.alpha_h(0)/ (self.alpha_h(0) + self.beta_h(0))
            print(f"m(0): {self.m} \n n(0): {self.n} \n h(0): {self.h}")
                    

            # EULER INTEGRATION
            for step in range(self.num_steps):
                # Store current values
                self.v_hist[step] = self.v
                self.m_hist[step] = self.m
                self.n_hist[step] = self.n
                self.h_hist[step] = self.h

                # Update value of voltage using Euler int.
                self.v += self.dt * self.grad_v(step)
                self.m += self.dt * self.grad_m(step)
                self.n += self.dt * self.grad_n(step)
                self.h += self.dt * self.grad_h(step)

            if plot:

                plt.figure(figsize=(20,10))
                plt.plot(np.array(range(self.num_steps))*0.001, self.v_hist)
                plt.title(fr"$I = {self.i_ext}$" + " (mA/nF)")
                plt.xlabel(r"$t$")
                plt.ylabel(r"$v$")
                plt.ylim([-80,50])
                plt.savefig(f"mem_pot_sim1_{i}.png", bbox_inches='tight', dpi=300)

                plt.figure(figsize=(20,10))
                plt.plot(np.array(range(self.num_steps))*0.001, self.m_hist)
                plt.xlabel(r"$t$")
                plt.ylabel(r"$m$")
                plt.ylim([0,1])
                plt.savefig(f"m_sim1_{i}.png", bbox_inches='tight', dpi=300)

                plt.figure(figsize=(20,10))
                plt.plot(np.array(range(self.num_steps))*0.001, self.n_hist)
                plt.xlabel(r"$t$")
                plt.ylabel(r"$n$")
                plt.ylim([0,1])
                plt.savefig(f"n_sim1_{i}.png", bbox_inches='tight', dpi=300)

                plt.figure(figsize=(20,10))
                plt.plot(np.array(range(self.num_steps))*0.001, self.h_hist)
                plt.xlabel(r"$t$")
                plt.ylabel(r"$h$")
                plt.ylim([0,1])
                plt.savefig(f"h_sim1_{i}.png", bbox_inches='tight', dpi=300)
            
            above_t = (self.v_hist > 20).astype(int)
            cross_t = np.diff(above_t) == 1
            num_spikes = sum(cross_t)
            if num_spikes < 2:
                num_spikes = 0
            
            frequencies[num] = num_spikes * 5
            
        plt.figure(figsize=(20,10))
        plt.plot(i_ext, frequencies)
        plt.xlabel(r"$I$" + " (mA/nF)")
        plt.ylabel("Spiking frequency (Hz)")
        #plt.savefig("spike_freq_hh.png", bbox_inches='tight', dpi=300)
                     
    def simulate_2(self, I=2.3, plot=True):
        
        i_ext = self.generate_input(I=I)
        
        self.v = -65
        self.m = 0
        self.h = 1
        self.n = 0
        self.i_ext = i_ext
        self.num_steps = len(self.i_ext)
        self.v_hist = np.zeros((self.num_steps))
        self.m_hist = np.zeros((self.num_steps))
        self.n_hist = np.zeros((self.num_steps))
        self.h_hist = np.zeros((self.num_steps))
        
        # Initialise resting conditions for gating variables
        self.v_hist[0] = self.v
        self.m = self.alpha_m(0)/ (self.alpha_m(0) + self.beta_m(0))
        self.n = self.alpha_n(0)/ (self.alpha_n(0) + self.beta_n(0))
        self.h = self.alpha_h(0)/ (self.alpha_h(0) + self.beta_h(0))

        # EULER INTEGRATION
        for step in range(self.num_steps):
            # Store current values
            self.v_hist[step] = self.v
            self.m_hist[step] = self.m
            self.n_hist[step] = self.n
            self.h_hist[step] = self.h

            # Update value of voltage using Euler int.
            self.v += self.dt * self.grad_v(step, var=True)
            self.m += self.dt * self.grad_m(step)
            self.n += self.dt * self.grad_n(step)
            self.h += self.dt * self.grad_h(step)

        if plot:

            plt.figure(figsize=(20,10))
            plt.plot(np.array(range(self.num_steps))*0.001, self.v_hist)
            plt.xlabel(r"$t$" + " (ms)")
            plt.ylabel(r"$v$" + " (mV)")
            plt.savefig(f"variable_mem_{I}.png", bbox_inches='tight', dpi=300)

            plt.figure(figsize=(20,10))
            plt.plot(np.array(range(self.num_steps))*0.001, self.m_hist)
            plt.plot(np.array(range(self.num_steps))*0.001, self.n_hist)
            plt.plot(np.array(range(self.num_steps))*0.001, self.h_hist)
            plt.xlabel(r"$t$" + " (ms)")
            plt.ylabel(r"$m,n,h$")
            plt.legend(["m", "n", "h"])
            plt.ylim([0,1])
            plt.savefig(f"variable_gates_{I}.png", bbox_inches='tight', dpi=300)
            
            plt.figure(figsize=(20,10))
            plt.plot(np.array(range(self.num_steps))*0.001, self.i_ext)
            plt.xlabel(r"$t$" + " (ms)")
            plt.ylabel(r"$I$" + " (mA/nF)")
            plt.savefig(f"variable_pulse_{I}.png", bbox_inches='tight', dpi=300)
    
    
    
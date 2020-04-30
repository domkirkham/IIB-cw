import numpy as np
import random
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import seaborn as sns

class Population:
    """Abstract class for populations"""
    def __init__(self, weights, N=1000, K=100, v_th=1, dt=0.1e-3, tau=20e-3, rx=10):
        self.pop_size = N
        self.input_size = K
        self.v_th = v_th
        self.dt = dt
        self.tau = tau
        self.rx = rx
        self.weights = weights

        
class Excitatory(Population):
    def __init__(self, weights, N=1000, K=100, rx=10):
        super().__init__(weights=weights, N=N, K=K, rx=rx)
        cons_array = np.zeros((self.pop_size, 3, self.pop_size))
        # Generate the random connections
        for i, neuron in enumerate(cons_array):
            for j, pop in enumerate(neuron):
                cons_array[i,j,random.sample(range(self.pop_size), self.input_size)] = 1
        self.inputs = cons_array
        self.inputs_idx = np.where(self.inputs == 0)
        print(np.sum(self.inputs))
        
    def init_dynamics(self, t, weights):
        # Initialise activity tracking matrix
        self.v = np.zeros((int(t//self.dt), self.pop_size))
        self.activity = np.zeros((int(t//self.dt), self.pop_size))
        
    def update_activity(self, network_activity, k, reset=False, multip=False, full=False):
        """
        inputs:
        network_activity: array, shape (3,N), current network voltages for all neurons"""
        # Ignore activity from non input neurons
        
        input_activity = network_activity.copy()
        
        if multip:
            for i in range(self.pop_size):
                input_activity[i,:,:int(self.input_size/2)] = -input_activity[i,:,:int(self.input_size/2)]
                
        input_activity[self.inputs_idx] = 0 
        input_activity = np.matmul(self.weights[0], input_activity)
        input_activity = np.sum(input_activity, axis=1)
            
        self.v[k] = self.v[k-1] + (self.dt * ((-self.v[k-1]/self.tau) + (input_activity/np.sqrt(self.input_size))))
        
        # Update spiking
        spike_neurons = np.where(self.v[k] > self.v_th)
        self.activity[k, spike_neurons] = 1 / self.dt
        # Reset voltages for spiking neurons
        if reset:
            self.v[k, spike_neurons] = 0
        
    
class Inhibitory(Population):
    def __init__(self, weights, N=1000, K=1000, rx=10):
        super().__init__(weights=weights, N=N, K=K, rx=rx)
        cons_array = np.zeros((self.pop_size, 3, self.pop_size))
        # Generate the random connections
        for i, neuron in enumerate(cons_array):
            for j, pop in enumerate(neuron):
                cons_array[i,j,random.sample(range(self.pop_size), self.input_size)] = 1
        self.inputs = cons_array
        self.inputs_idx = np.where(self.inputs == 0)
        print(np.sum(self.inputs))
        
    def init_dynamics(self, t, weights):
        # Initialise activity tracking matrix
        self.v = np.zeros((int(t//self.dt), self.pop_size))
        self.activity = np.zeros((int(t//self.dt), self.pop_size))
        
    def update_activity(self, network_activity, k, reset=True, multip=False, full=False):
        """
        inputs:
        network_activity: array, shape (3,N), current network voltages for all neurons"""
        input_activity = network_activity.copy()
        
        if multip:
            for i in range(self.pop_size):
                input_activity[i,:,:int(self.input_size/2)] = -input_activity[i,:,:int(self.input_size/2)]
                
        input_activity[self.inputs_idx] = 0 
        input_activity = np.matmul(self.weights[1], input_activity)
        input_activity = np.sum(input_activity, axis=1)
            
        self.v[k] = self.v[k-1] + (self.dt * ((-self.v[k-1]/self.tau) + (input_activity/np.sqrt(self.input_size))))
        
        # Update spiking
        spike_neurons = np.where(self.v[k] > self.v_th)
        self.activity[k, spike_neurons] = 1 / self.dt
        # Reset voltages for spiking neurons
        if reset:
            self.v[k, spike_neurons] = 0
        
        
        # Ignore activity from non input neurons
        #for i in range(self.pop_size):
            #input_activity = network_activity.copy()
            #input_activity[np.where(self.inputs[i] == 0)] = 0
            #if multip:
           #     input_activity[:,:int(self.input_size/2)] = -input_activity[:,:int(self.input_size/2)]
            # Weight activities
           # input_activity = np.sum(np.matmul(self.weights[1], input_activity))
            # Update V
           # if multip:
           #     self.v[k,i] = self.v[k-1,i] + (self.dt * ((-self.v[k-1,i]/self.tau) + (input_activity/np.sqrt(self.input_size))))
           # elif full:
           #     self.v[k,i] = self.v[k-1,i] + (self.dt * ((-self.v[k-1,i]/self.tau) + (input_activity/np.sqrt(self.input_size))))
          #  else:
           #     self.v[k,i] = self.v[k-1,i] + (self.dt * ((-self.v[k-1,i]/self.tau) + (input_activity/self.input_size)))
            # Update spiking activity
          #  if self.v[k,i] > self.v_th:
          #      self.activity[k,i] = 1 / self.dt
                # Reset voltages for spiking neurons
          #      if reset:
          #          self.v[k,i] = 0
          #  if multip:
          #      break
        

class External(Population):
    def __init__(self, N=1000, K=100, rx=10):
        super().__init__(weights=None, N=N, K=K, rx=rx)
        
    def sample_activity(self, t=2, plot=False):
        self.num_samples = int(t // self.dt)
        activity = (1/self.dt) * bernoulli.rvs(p = self.rx * self.dt, size = (self.num_samples, self.pop_size))
        self.activity = activity
        
        if plot:
            
            avg_firing_rate = self.dt * np.sum(np.mean(activity, axis=1)) / t
            plt.figure(figsize=(20,40))
            #print(np.where(self.e_pop.activity > 1))
            spikes_plot = []
            for neuron in activity.T:
                spikes_plot.append(np.where(neuron > 1)[0].T)
            plt.eventplot(spikes_plot, color='k', linelengths = 1, linewidths=0.5)
            sns.despine(top=True, right=True, bottom=True, left=True)
            plt.xlim([0,len(activity)])
            #plt.eventplot(neuralData, color=colorCodes, linelengths = lineSize) 
            #plt.imshow(self.e_pop.v.T, origin='lower')
            plt.xlabel(r"$k$")
            plt.yticks([])
            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelbottom=False)
            
            
            
            print(f"Average firing rate: {avg_firing_rate}")
            print(f"True firing rate: {self.rx}")
            
            #plt.title(f"Raster plot for external neurons, average firing rate {avg_firing_rate} Hz")
            plt.savefig(f"external_pop_{t}.png", dpi=300, bbox_inches='tight')
        
        
        
        
        
        
        

        
        
  
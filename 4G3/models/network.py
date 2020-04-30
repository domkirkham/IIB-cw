import numpy as np
from .population import Population, Excitatory, Inhibitory, External
from tqdm import tqdm
import matplotlib.pyplot as plt

class Network:
    """Network class"""
    def __init__(self, N=1000, K=100, weights=np.array([[0,0,1],[0,0,0]]), rx=10):
        self.pop_size = N
        self.input_size = K
        self.weights = weights
        self.rx = rx
        self.e_pop = Excitatory(weights=self.weights, N=self.pop_size, K=self.input_size, rx=self.rx)
        self.i_pop = Inhibitory(weights=self.weights, N=self.pop_size, K=self.input_size, rx=self.rx)
        self.x_pop = External(N=self.pop_size, K=self.input_size, rx=self.rx)
        
    def simulate(self, t, plot=False, reset=False, multip=False, full=False):
        
        self.x_pop.sample_activity(t=t, plot=plot)
        self.e_pop.init_dynamics(t=t, weights=self.weights)
        self.i_pop.init_dynamics(t=t, weights=self.weights)
        
        for k in tqdm(range(1, self.x_pop.num_samples)):
            network_activity = np.vstack((self.e_pop.activity[k-1], self.i_pop.activity[k-1], self.x_pop.activity[k-1]))
            network_activity = np.tile(network_activity, (self.e_pop.pop_size,1,1))
            self.e_pop.update_activity(network_activity, k, reset=reset, multip=multip, full=full)
            self.i_pop.update_activity(network_activity, k, reset=reset, multip=multip, full=full)
            
        if plot:
            plt.figure(figsize=(20,5))
            plt.subplot(2,1,1)
            plt.plot(range(self.x_pop.num_samples), self.e_pop.v[:,0], lw=1.5)
            plt.plot(range(self.x_pop.num_samples), self.i_pop.v[:,0], lw=1.5)
            plt.ylabel(r"$V_{i}$")
            plt.xlim([0,len(self.e_pop.activity)])
            plt.xticks([])
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
            
            
            #plt.figure(figsize=(20,10))
            #plt.plot(range(self.x_pop.num_samples), self.i_pop.v[:,0])
            
            
            
            #plt.figure(figsize=(20,5))
            plt.subplot(2,1,2)
            #print(np.where(self.e_pop.activity > 1))
            plt.eventplot([np.where(self.e_pop.activity > 1)[0], np.where(self.i_pop.activity > 1)[0], np.where(self.x_pop.activity > 1)[0]], color='k', linelengths = 0.2, linewidths=[1,1,1])
            plt.xlim([0,len(self.e_pop.activity)])
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
            
            if multip:
                avg_firing_rate_e = self.e_pop.dt * np.sum(self.e_pop.activity[:,0]) / t
                avg_firing_rate_i = self.i_pop.dt * np.sum(self.i_pop.activity[:,0]) / t
            else:
                avg_firing_rate_e = self.e_pop.dt * np.sum(np.mean(self.e_pop.activity, axis=1)) / t
                avg_firing_rate_i = self.i_pop.dt * np.sum(np.mean(self.i_pop.activity, axis=1)) / t
            
            print(f"Average firing rate (E): {avg_firing_rate_e}")
            print(f"Average firing rate (I): {avg_firing_rate_i}")
            print(f"True firing rate: {self.e_pop.rx}")
            
            #plt.title(f"Raster plot for external neurons, average firing rate {avg_firing_rate} Hz")
            plt.tight_layout
            plt.savefig(f"e_pop_v_{t}_{self.e_pop.input_size}.png", dpi=300, bbox_inches='tight')
            #plt.colorbar()
            
        mean = np.mean(self.e_pop.v[int(0.1//self.e_pop.dt):, 0])
        var = np.var(self.e_pop.v[int(0.1//self.e_pop.dt):, 0])
        
        counts = []
        window_length = int(0.1//self.e_pop.dt)
        if multip:
            for window in range(int(self.x_pop.num_samples//window_length)-1):
                counts.append(np.sum(self.e_pop.activity[window*window_length:(window+1)*window_length,0])*self.e_pop.dt)
        else:
            for window in range(int(self.x_pop.num_samples//window_length)-1):
                counts.append(np.sum(self.e_pop.activity[window*window_length:(window+1)*window_length,0])*self.e_pop.dt)
            
        print(f"Estimated mean: {mean} \n Estimated variance: {var} \n Estimated Fano factor: {np.var(counts)/np.mean(counts)}")
        
            
            
            
            
            
        
        

        
  
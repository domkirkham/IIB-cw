import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import random
from scipy.stats import norm

class Network:
    """Network class"""
    def __init__(self, N=1000):
        """Initialise neuron activity as bool"""
        self.pop_size = N
        self.activity = np.zeros((self.pop_size, 1), dtype='bool')
        self.weights = np.zeros((self.pop_size, self.pop_size))
        
    def simulate(self, M, k, noise_level=0, error=False):
        
        self.num_patterns = M
        self.num_steps = k
        self.act_hist = np.zeros((self.pop_size, self.num_steps), dtype='bool')
        self.weight_hist = np.zeros((self.pop_size, self.pop_size, self.num_steps), dtype='float32')
        self.errors = []
        
        # We want to store a set of M states or patterns
        # In each pattern, each bit has a 0.5 prob of being 1
        # The patterns have the same length and the number of neurons
        self.patterns = np.random.binomial(1, 0.5, (self.num_patterns, self.pop_size))
        
        # WEIGHT UPDATE
        # reset weights to 0
        self.weights = np.zeros((self.pop_size, self.pop_size))
        # loop the sum over patterns
        print("Forming weight matrix...")
        for p in range(self.num_patterns):
            #act_matrix_i = np.tile(self.patterns[p], (self.pop_size,1)).astype("float")
            #act_matrix_j = np.tile(self.patterns[p], (self.pop_size,1)).T.astype("float")
            self.weights += np.outer((self.patterns[p] - 0.5), (self.patterns[p] - 0.5))
        np.fill_diagonal(self.weights, 0)
        
        # ASYNC ACTIVITY UPDATE
        print("Testing memory recall...")
        for mem in tqdm(self.patterns):
            self.activity = mem.copy()
            if error:
                self.activity_error = mem.copy()
            if noise_level > 0:
                noise = bernoulli.rvs(p = noise_level, size = self.pop_size)
                flip_locs = np.where(noise == 1)
                self.activity[flip_locs] = 1 - self.activity[flip_locs]
                self.activity_error[flip_locs] = 1 - self.activity_error[flip_locs]
            for step in range(self.num_steps):
                neurons = list(range(self.pop_size))
                random.shuffle(neurons)
                for k in neurons:
                    # update current activity
                    self.activity[k] = np.dot(self.weights[k], self.activity) >= 0
                # Single step error rate
            self.activity_error = np.dot(self.weights, self.activity_error) >= 0

            self.errors.append(np.sum(self.activity_error != mem))
            
        t_error = norm.cdf(0, loc=(self.pop_size-1)/8, scale=np.sqrt((self.num_patterns-1+1e-8)*(self.pop_size-1+1e-8)/32))
            
        print(f"Average error percentage: {100*np.sum(self.errors)/(self.pop_size*self.num_patterns)}")
        print(f"Theoretical error percentage: {100*t_error}")
        return np.sum(self.errors)/(self.pop_size*self.num_patterns), t_error
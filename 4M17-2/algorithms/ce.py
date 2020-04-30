import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from tqdm import tqdm
import time


class CEOptimiser:
    """Implements the cross-entropy method for optimisation"""
    def __init__(self, pop_size=100, dims=2, init_type='normal', init_mean=0, init_var=1, obs_var=0):
        """
        Initialises all the required fields for the problem
        args:
        pop_size (int): How many samples to draw from the distribution
        dims (int): Dimension of the problem
        init_type (str): Distribution to draw initial population from (currently normal only)
        init_mean (float): Mean of distribution at initialisation
        init_var (float): Variance of distribution at initialisation
        """

        self.dims = dims
        self.pop_size = pop_size
        self.obs_var = obs_var
        self.fitness_log = []
        self.var_log = []
        self.mean_log = []

        # initialise population by drawing from chosen distribution
        if init_type == 'normal':
            self.mean = np.zeros(self.dims)
            self.var = init_var * np.eye(dims)
            self.pop_init = np.random.multivariate_normal(self.mean, self.var, self.pop_size)
        elif init_type == 'uniform':
            self.pop_init = np.zeros((dims, self.pop_size))
            self.mean = init_mean
            self.var = init_var  
            self.pop_init = np.random.multivariate_normal(self.mean, self.var, self.pop_size)

        self.pop = self.pop_init
    
    def optimise(self, num_evals=30, plot=False, weighted=False, sample_factor=7, alpha=0.7, tol=0.1):
        """
        Main method for evaluating the objective function

        args:
        num_evals (int): Number of generations
        plot (bool): Whether to plot the population and a heatmap of the function
        weighted (bool): Whether to use fitness-weighted intermediate recombination
        sample_factor (int): as in _resample
        alpha (float, [0,1]): Smoothing factor for distribution update
        """
        
        self.timings = []
        
        for ev in tqdm(range(num_evals)):
            self.eval = ev+1
            t = time.process_time()
            self._resample(sample_factor=sample_factor)  
            self.timings.append(time.process_time() - t)
            if plot:
                if self.eval % 10 == 0 or ev == 0:
                    self._plot()
            if self.eval > 1:
                if abs(self.fitness_log[-1] - self.fitness_log[-2]) < tol:
                    self.opt_time = self.eval
                    break
    
    
    def _resample(self, sample_factor=7, alpha=0.7):
        """
        1. Evaluate function on top samples
        2. Update distribution parameters based on these
        3. Sample new population from updates distribution

        args:
        sample_factor (int): selection size/population size. e.g. if 7, choose top 1/7th of samples
        alpha (float, [0,1]): smoothing factor"""
        
    
        num_samples = self.pop_size // sample_factor
        
        self.fitness = self._shubert(self.pop)
        
        self.fitness_log.append(np.mean(self.fitness))
        # Choose top samples
        chosen_samples = np.argsort(self.fitness)[:num_samples]
        # Update parameters 
        self.mean = (1-alpha) * self.mean + alpha * np.mean(self.pop[chosen_samples], axis=0)
        self.mean_log.append(self.mean)
        # Keep the Gaussian symmetric
        self.var = (1-alpha) * self.var + alpha * np.diag(np.var(self.pop[chosen_samples], axis=0))
        self.var_log.append(np.diag(self.var))
        #self.var = (1-alpha) * self.var + alpha * np.cov(self.pop[chosen_samples].T)
        # Resample population
        self.pop = np.random.multivariate_normal(self.mean, self.var, self.pop_size)
        
    def _plot(self):
        """
        Plots a heatmap of the objective function in the range [-2,2], overlays current population
        """
        plt.figure(figsize=(12,10))
        # Make data grid for heatmap
        x_range = np.linspace(-2,2,100)
        x1, x2 = np.meshgrid(x_range, x_range)
        positions = np.vstack([x1.ravel(), x2.ravel()]).T
        z = shubert(positions).reshape(-1,100)
        plt.pcolormesh(x1, x2, z, cmap=cm.coolwarm)
        plt.title("Timestep " + str(self.eval))
        # plt.colorbar()
        plt.xlim([-3,3])
        plt.ylim([-3,3])
        # Plot current population
        plt.scatter(self.pop.T[0], self.pop.T[1], c='k', s=15)
    
    def _shubert(self, x):
        """
        Takes input array with the points at which to evaluate the function. Value becomes -Inf if outside the domain.
        args: 
        x (ndarray) : dim*num_points array of points

        returns: 
        list of (noisy) values at desired points
        """
        dim = x.shape[1]
        num_samples = x.shape[0]
        vals = []
        
        for sample in x:
            penalty = 0
            # Check sample is in domain
            if max(sample) > 2:
                penalty += self.eval**3 * (max(sample)-2)
            if min(sample) < -2:
                penalty -= self.eval**3 * (min(sample)+2)

            i_array = np.tile(sample, (5,1))
            j_array = np.tile(range(5), (dim,1)).T + 1

            assert i_array.shape == j_array.shape

            val_array = j_array * np.sin((j_array+1)*i_array + j_array)

            vals.append(np.sum(np.sum(val_array, axis=1), axis=0) + penalty)

        return np.array(vals) + np.sqrt(self.obs_var)*np.random.randn(num_samples)
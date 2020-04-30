import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from tqdm import tqdm
import time


class ESPopulation:
    """This class implements an evolution strategy to find the minimum of the Schubert function of a given 
    dimension."""
    
    def __init__(self, pop_size=50, dims=2, init_type='normal', obs_var=0):
        """
        Initialises all the required fields for the problem
        args:
        pop_size (int): Size of the population
        dims (int): Dimension of the problem
        init_type (str): Distribution to draw initial population from (currently normal only)
        obs_var (float): Variance of noise added to observation of function
        
        Example usage: 10 fold CV, logging population mean fitness (see Figure 5)
        >>>> fitness = np.zeros((60,10))
        >>>> for i in range(10):
        >>>>     test = ESPopulation(pop_size=200, dims=5)
        >>>>     test.optimise(num_evals=60, plot=False, weighted=True, tol=0)
        >>>>     fitness[:len(test.fitness_log),i] = test.fitness_log
        """
        
        self.pop_size = pop_size
        self.obs_var = obs_var
        self.dims = dims
        self.fitness_log = []
        self.var_log = []
        # covariances and rotation angles for every sample
        self.cov = np.dstack([np.eye(self.dims)] * pop_size)
        self.rot = np.dstack([np.zeros((self.dims, self.dims))] * pop_size)
        self.mean = np.zeros(dims)
        # Use Schwefel [1987] recommendation of 1:7 mu:lambda ratio
        self.num_parents = self.pop_size // 7
        # Use Schwefel [1995] recommendation for control paramters
        self.tau = 1 / (np.sqrt(2*np.sqrt(self.dims)))
        self.tau_prime = 1 / (np.sqrt(2*self.dims))
        self.beta = 0.0873

        # initialise population by drawing from chosen distribution
        if init_type == 'normal':
            self.pop_init = np.random.multivariate_normal(self.mean, self.cov[:,:,0], self.pop_size)
        else:
            self.pop_init = np.zeros((dims, self.pop_size))
        self.pop = self.pop_init
        
        # Form the initial rotation angle matrix
        self._cov_to_rot()
    
           
    def assess_pop(self):
        """
        Assesses the current population:
        1. Removes/thresholds current solutions in population outside of range
        2. Assesses the target function on this 'fixed' population
        3. Chooses parents for next generation
        """
        out_range = []
         #This thresholds values outside the range
        for i, sample in enumerate(self.pop):
            if max(sample) > 2 or min(sample) < -2:
                # Threshold the out of range samples OR
                #self.pop[i][np.where(sample > 2)] = 2
                #self.pop[i][np.where(sample < -2)] = -2
                # Remove options out of range (as suggested in handouts)
                out_range.append(i)
        
        # Delete entries corresponding to out of range solutions
        fixed_pop = np.delete(self.pop, out_range, 0)
        fixed_cov = np.delete(self.cov, out_range, 2)
        fixed_rot = np.delete(self.rot, out_range, 2)
                
        # Evaluate fitness on 'fixed' population
        self.fitness = self._shubert(fixed_pop)
        self.fitness_log.append(np.mean(self.fitness))
        # Choose parents as top num_parents
        parent_ids = np.argsort(self.fitness)[:self.num_parents]
        # Form parent values for weights (negative since we minimise)
        self.parent_values = [-self.fitness[i] for i in parent_ids]
        # Weights according to fitness, cast to [0,1]
        self.parent_weights = (self.parent_values - min(self.parent_values)) / (sum(self.parent_values - min(self.parent_values)) + 1e-8)
        self.parents = np.array([fixed_pop[i] for i in parent_ids])
        self.parents_cov = np.array([fixed_cov[:,:,i] for i in parent_ids])
        self.parents_rot = np.array([fixed_rot[:,:,i] for i in parent_ids])
        
    def generate_offspring(self, weighted=False):
        """
        Generate offspring from the current parents, updating solutions and strategy parameters.
        - Uses discrete recombination on the control variables and intermediate recombination on strategy parameters,
        as suggested in handouts.
        - Intermediate recombination is weighted by the fitness of each parent
        
        args:
        weighted (bool): Whether to use fitness-weighted intermediate recombination
        """
        # Form covariance and rotation angle matrices for offspring
        off_cov = np.zeros(self.cov.shape)
        off_rot = np.zeros(self.rot.shape)
        # Value in array gives the parent that this value comes from
        num_parents = min(self.num_parents, len(self.parents))
        # Array of parent indices for each offspring
        off_parents = np.random.randint(0, num_parents, size=(self.pop_size, 2))
        # 0 or 1 based on which parent the control variable comes from
        off_inherit = np.random.randint(0, 2, size=(self.pop_size, self.dims))
        # Add control variables from respective parents to new array
        off_cont = np.zeros(off_inherit.shape)
        for i, off in enumerate(off_inherit):
            for j, dim in enumerate(off):
                off_cont[i, j] = self.parents[off_parents[i, dim], j]
        
        # Intermediate recombination for strategy parameters
        if weighted:
            
            for i, off in enumerate(off_parents):
                weights = self.parent_weights[off] / (sum(self.parent_weights[off]) + 1e-8)
                off_cov[:, :, i] =  weights[0] * self.parents_cov[off[0]] + weights[1] * self.parents_cov[off[1]]
                off_rot[:, :, i] = weights[0] * self.parents_rot[off[0]] + weights[1] * self.parents_rot[off[1]]

        else:
            for i, off in enumerate(off_parents):
                off_cov[:, :, i] = 0.5 * self.parents_cov[off[0]] + 0.5 * self.parents_cov[off[1]]
                off_rot[:, :, i] = 0.5 * self.parents_rot[off[0]] + 0.5 * self.parents_rot[off[1]]

        # Replace cov and rot for new population
        self.cov = off_cov
        self.rot = off_rot
        # Update parameters in each
        self._rot_to_cov()
        self._cov_to_rot()
        # Update current population
        self.pop = off_cont
        
    def mutate(self):
        """
        Carries out mutation of strategy parameters, then control variables (as in notes)
        """
        for n in range(self.pop_size):
            # First we mutate the strategy parameters for the whole population
            n_zero = np.random.randn(1)
            n_one = np.random.randn(self.cov.shape[0])
            # mutate diagonal
            self.cov[:,:,n][np.diag_indices_from(self.cov[:,:,n])] *= np.exp(2*self.tau_prime*n_zero + 2*self.tau*n_one + 1e-8)
            # mutate rotation angles
            self.rot[:,:,n] += self.beta * np.random.randn(self.rot.shape[0],self.rot.shape[1])
                    
        # update rotation angle and covariance matrices
        self._rot_to_cov()
        self._cov_to_rot()
        
        # Trace mean variances for each dimension
        self.var_log.append(np.diag(np.mean(self.cov, axis=2)))
        
        # mutate the control variables for each member of population
        for n in range(self.pop_size):
            self.pop[n] += np.random.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov[:,:,n])
        
    def optimise(self, num_evals=10, plot=False, weighted=False, tol=0.1):
        """
        Main method for evaluating the objective function
        
        args:
        num_evals (int): Number of generations
        plot (bool): Whether to plot the population and a heatmap of the function
        weighted (bool): Whether to use fitness-weighted intermediate recombination
        tol (float): tolerance for stopping criteron -- smallest allowable change in mean population fitness
        """
        
        self.timings = []
        
        for ev in tqdm(range(num_evals)):
            self.eval = ev + 1
            t = time.process_time()
            self.assess_pop()
            self.generate_offspring(weighted=weighted)
            self.mutate()
            self.timings.append(time.process_time() - t)
            if plot:
                if self.eval % 5 == 0 or ev == 0:
                    self._plot()
            if self.eval > 1:
                if abs(self.fitness_log[-1] - self.fitness_log[-2]) < tol:
                    self.opt_time = self.eval
                    break
                              
    def _plot(self):
        """
        Plots a heatmap of the objective function in the range [-2,2], overlays current population
        """
        plt.figure(figsize=(12,10))
        # Make data grid for heatmap
        x_range = np.linspace(-2,2,100)
        x1, x2 = np.meshgrid(x_range, x_range)
        positions = np.vstack([x1.ravel(), x2.ravel()]).T
        z = self._shubert(positions).reshape(-1,100)
        plt.pcolormesh(x1, x2, z, cmap=cm.coolwarm)
        #plt.colorbar()
        plt.title("Timestep " + str(self.eval))
        # plt.colorbar()
        plt.xlim([-3,3])
        plt.ylim([-3,3])
        # Plot current population
        plt.scatter(self.pop.T[0], self.pop.T[1], c='k', s=15)
        plt.savefig("es_" + str(self.pop_size) + "_" + str(self.eval) + ".png", bbox_inches='tight', dpi=200)
        
    def _cov_to_rot(self):
        """
        Takes current values of covariance matrix and updates rotation angle matrix
        """
        # form matrix of rotation angles
        for n in range(self.rot.shape[2]):
            for i, row in enumerate(self.rot[:,:,n]):
                for j, val in enumerate(row):
                    if not i == j:
                        self.rot[i,j,n] = 0.5 * np.arctan(2 * self.cov[i,j,n] / (self.cov[i,i,n]-self.cov[j,j,n] + 1e-8))
        
    def _rot_to_cov(self):
        """
        Takes current values of rotation angle matrix and updates covariance matrix
        """
        for n in range(self.cov.shape[2]):
            for i, row in enumerate(self.cov[:,:,n]):
                for j, val in enumerate(row):
                    if not i == j:
                        self.cov[i,j,n] = 0.5 * (self.cov[i,i,n]-self.cov[j,j,n]) * np.tan(2 * self.rot[i,j,n] + 1e-8)
        
        
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
    
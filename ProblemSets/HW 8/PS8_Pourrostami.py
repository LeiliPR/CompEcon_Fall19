# DETERMINISTIC INFINITE PERIOD MODEL

#####################################################

# 1. No Government Contract:
    
# IMPORT PACKAGES

# To Display multiple outputs from a given cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Importing Libraries
import numpy as np
import pandas as pd
from numpy import random
from scipy.optimize import fminbound
from scipy.optimize import minimize
from scipy.optimize import brentq, fsolve
from scipy.interpolate import interp1d
from numba import njit,jit, vectorize

# Setting up Matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

class Euler_Infinite_Period_Deterministic_Agent_Model:
    """
    This class contains the parameters of Deterministic Dynamic Programming Squared Model in Infinite Periods
    with no government(principal) role. It solves for optimal k value for a given array of state value(y_grid).
    It uses scipy BRENTQ algorithm to solve for k_{t+1} in euler equation given k_{t}
    """
    def __init__(self,
                 β = 0.9,
                 α = 0.9,
                 δ = 0.5,
                 θ = 0.5,
                 a = 10,
                 b = 5,
                 y_min = 1e-5,
                 y_max = 5,
                 y_grid_size = 100
                ):
        """
        Parameters are defined as below
        --------------------------------
        # Model Parameters
        α: relates the y(t) with y(t-1)
        δ: power of k in state function G_t(y_t,x_t,k_t)
        θ: power of x in state function G_t(y_t,x_t,k_t)
        a: coeff. of x in state function G_t(y_t,x_t,k_t)
        b: coeff. of k in state function G_t(y_t,x_t,k_t)
        β: discount factor of agent utility
        
        # State Paramters
        y_min: minimum y (state) value to consider 
        y_max: maximum y (state) value to consider 
        y_grid_size: No of points to select between y_min and y_max
    
        """
    
        # Model Parameters
        self.β = β
        
        # State Function G Parameters assignement
        self.α, self.δ, self.θ, self.a, self.b = α, δ, θ, a, b
        
        #Y Grid State Allocation
        self.y_min, self.y_max, self.y_grid_size  = y_min, y_max, y_grid_size
        self.y_grid = np.linspace(self.y_min, self.y_max, self.y_grid_size)
    
    def next_y(self,y,x,k):
        """
        State Function G_t(y_t,x_t,k_t)
        
        """
        return y**self.α + self.a * x**self.θ + self.b * k**self.δ

    def p_y_k(self,k):
        """
        Partial derivative of y_{t+1} wrt k_t
        """
        return self.b * self.δ/( k**( 1-self.δ ) )

    def p_y_y(self,y):
        """
        Partial derivative of y_{t+1} wrt y_t
        """
        return self.α/y**(1-self.α)

    def h_prime(self,y,k):
        """
        Partial Derivative of h_t wrt to c_t;
        """
        return 1/(y-k)

    def coleman_operator(self, k_grid):
        """
        This function apply coleman_operator on on input policy function 
        and outpus policy function on k. It defines an euler equation which 
        relates k_{t+1} with k_t
        """
        kg = np.zeros_like(k_grid)
        g_func = lambda x: np.interp(x, self.y_grid, k_grid)
        for i, y in enumerate(self.y_grid):
            def euler_equation(k):
                y_next = self.next_y(y,0,k)
                k_next = g_func(y_next)
                LHS = self.h_prime(y,k)
                RHS = ( 
                       self.β * self.p_y_k(k) * self.h_prime(y_next,k_next) * 
                       (1 + self.p_y_y(y_next)/self.p_y_k(k_next) )
                       )
                return LHS - RHS   
            #k_star = fsolve(euler_equation,x0 = y/2)
            k_star = brentq(euler_equation,1e-10,y-1e-10)
            kg[i] = k_star
        return kg


    def solve_policy_function(self,
                              tol = 1e-5,
                              max_iter = 500,
                             show_graph = True):
        """
        This function solves the complete class Euler_Infinite_Period_Deterministic_Agent_Model
        
        """
        k_grid = self.y_grid/5
        
        if show_graph:
            fig, ax = plt.subplots(figsize=(10, 6))
            lb = 'initial condition $k(y) = 0$'
            ax.plot(self.y_grid, k_grid, color = 'red', lw=2, alpha=0.6, label=lb);

        i = 0
        error = tol + 1

        while error > tol and i < max_iter:
            new_k_grid = self.coleman_operator(k_grid)
            error = np.max( np.abs(new_k_grid - k_grid) )
            k_grid = new_k_grid
            if show_graph:
                ax.plot(self.y_grid, k_grid, color=plt.cm.jet(i / 50), lw=2, alpha=0.6);
            i = i+1

        self.k_grid = k_grid
        if show_graph:
            lb = 'last policy iteration {}'.format(i)
            ax.plot(self.y_grid, self.k_grid, 'k-', lw=2, alpha=0.8, label=lb);
            ax.legend(loc='upper left')
            ax.set_ylabel('K')
            ax.set_xlabel('Y ')
        #plt.show();  
        
    def interpolate_k(self,y):
        """
        This function returns the optimal value of k_t for a given state value y_t
        """
        return np.interp(y,self.y_grid,self.k_grid)
 

 # Solve and Run the Model
EIAM = Euler_Infinite_Period_Deterministic_Agent_Model(y_max = 300,y_grid_size= 100)
EIAM.solve_policy_function()
EIAM.interpolate_k(100)

#################################################

# 2. With Government Contract

class Euler_Infinite_Period_Deterministic_Principal_Model:
    """
    This class contains the parameters of Deterministic Dynamic Programming Squared Model in Infinite Periods
    with government(principal) role. It solves for optimal k value for a given array of state value(y_grid).
    It uses scipy FSOLVE algorithm to solve for k_{t+1} and k_{t+1} in euler equation given k_{t},x_{t} and y_{t}
    """
    def __init__(self,
                 β = 0.9,
                 τ = 0.3,
                 α = 0.9,
                 δ = 0.5,
                 θ = 0.5,
                 a = 10,
                 b = 5,
                 y_min = 1e-5,
                 y_max = 200,
                 y_grid_size = 100,
                ):
        """
        Creates Instance of Dynamic Programming Squared Model based on Euler Equations
        --------------------------------
        Parameters are defined as below
        --------------------------------
        # Model Parameters
        α: relates the y(t) with y(t-1)
        δ: power of k in state function G_t(y_t,x_t,k_t)
        θ: power of x in state function G_t(y_t,x_t,k_t)
        a: coeff. of x in state function G_t(y_t,x_t,k_t)
        b: coeff. of k in state function G_t(y_t,x_t,k_t)
        β: discount factor of agent utility
        τ: tax rate of the government contract
        
        # State Paramters
        y_min: minimum y (state) value to consider 
        y_max: maximum y (state) value to consider 
        y_grid_size: No of points to select between y_min and y_max
    
        
        
        """
        
        # Model Parameters
        self.β, self.τ = β, τ
        
        # State Function G Parameters assignement
        self.α, self.δ, self.θ, self.a, self.b = α, δ, θ, a, b
        
        #Y Grid State Allocation
        self.y_min, self.y_max, self.y_grid_size  = y_min, y_max, y_grid_size
        self.y_grid = np.linspace(self.y_min, self.y_max, self.y_grid_size)
    
        
    def G(self,y,x,k):
        """
        State Function G_t(y_t,x_t,k_t)
        """
        return (y**self.α + self.a * x**self.θ + self.b * k**self.δ)#*self.shocks

    def p_y_k(self,k):
        """
        Partial derivative of y_{t+1} wrt k_t
        """
        return self.b * self.δ/( k**( 1-self.δ ) )

    def p_y_y(self,y):
        """
        Partial derivative of y_{t+1} wrt y_t
        """
        return self.α/y**(1-self.α)
    def p_y_x(self,x):
        """
        Partial derivative of y_{t+1} wrt x_t
        """
        return (self.a*self.θ)/ ( x**(1-self.θ) )

    # Principal Utility Partial Derivatives
    def p_f_x(self, y, x):
        """
        Partial derivative of f(y_t,x_t) wrt x_t
        """
        return -1/(self.τ*y - x)

    def p_f_y(self, y, x):
        """
        Partial derivative of f(y_t,x_t) wrt y_t
        """
        return self.τ/(self.τ*y - x)

    # Agent Utility Partial Derivatives
    def p_h_k(self, y, k):
        """
         Partial derivative of h(y_t,k_t) wrt k_t
        """
        return -1/((1-self.τ)*y-k)

    def p_h_y(self, y, k):
        """
        Partial derivative of h(y_t,k_t) wrt y_t
        """
        return (1-self.τ)/( (1-self.τ)*y-k )
    
    # Agent Value Function Partial Derivatives
    
    def hstar_prime(self, y, k):
        return 1/(y - k)
    
    def kstar_grid_func(self):
        """
        Creates a kstar grid values
        """
        eiam = Euler_Infinite_Period_Deterministic_Agent_Model(β = self.β ,
                                                               α = self.α,
                                                               δ =  self.δ,
                                                               θ = self.θ,
                                                               a = self.a,
                                                               b = self.b,
                                                               y_min = self.y_min, 
                                                               y_max =self.y_max, 
                                                               y_grid_size = self.y_grid_size)
        eiam.solve_policy_function(tol = 1e-6,show_graph = False)
        self.kstar_grid = eiam.k_grid
        
    def p_vstar_y(self, y): 
        """
        Partial Derviative of v*(y) wrt y
        """
        k = np.interp(y, self.y_grid, self.kstar_grid)
        return self.hstar_prime(y,k) * ( 1 + self.p_y_y(y)/self.p_y_k(k) )
    
    def p_v_y(self, y, k):
        """
        Partial Derviative of v(y) wrt y
        """
        return 1/self.β * 1/self.p_y_y(y) * ( self.p_vstar_y(y) - self.p_h_y(y,k) )

    # Contraints 
    def lambda_t(self, y, x, k):
        """
        Returns the values λ_t
        """
        return ( self.p_f_x(y,x)*self.p_y_k(k) )/( self.p_y_x(x) * self.p_h_k(y,k) )

    # Principal Value Function Partial Derivatives
    def p_w_y(self, y, x, k):
        """
        Parital Derivative of w(y_t) wrt y_t
        """
        return (
                self.p_f_y(y,x) 
               - ( self.p_f_x(y,x)*self.p_y_y(y) )/self.p_y_x(x) 
               + (self.p_f_x(y,x)*self.p_y_k(k))/(self.p_y_x(x)*self.p_h_k(y,k))*(self.p_h_y(y,k) - self.p_vstar_y(y) )
               )
    def x_foc(self, y_next, k_next, x_next, y, x, k):
        """
        Lagrangian FOC wrt to x
        """
        return (
                self.p_f_x(y,x) 
                + self.β * self.p_w_y(y_next,x_next,k_next) * self.p_y_x(x)
                + (self.lambda_t(y,x,k) * self.β * self.p_v_y(y,k) * self.p_y_x(x)  ) 
               )

    def k_foc(self, y_next, k_next, x_next, y, x, k):
        """
        Lagrangian FOC wrt to k
        """
        return ( 
                self.β * self.p_w_y(y_next,x_next,k_next) * self.p_y_k(k) 
                + self.lambda_t(y,x,k)
                * ( self.p_h_k(y,k) + self.β * self.p_v_y(y,k) * self.p_y_k(k) ) 
               )

    def coleman_operator_principal(self, k_grid, x_grid, tol = 1e-9):
        """
        Applies the coleman operator on k grid and x grid, tol is used in fsolve
        
        """
        g_k = k_grid
        g_x = x_grid

        g_k_func = lambda z: np.interp(z, self.y_grid, k_grid)
        g_x_func = lambda z: np.interp(z, self.y_grid, x_grid)

        def foc(pars,y):
            k,x = pars
            y_next = self.G(y,x,k)
            k_next = g_k_func(y_next)
            x_next = g_x_func(y_next)

            F = np.zeros(2)
            F[0] = self.k_foc(y_next, k_next, x_next, y, x, k)
            F[1] = self.x_foc(y_next, k_next, x_next, y, x, k)
            return F 

        for i, y in enumerate(self.y_grid):
            ans = fsolve(foc,
                         x0 =[ (1-self.τ)*y*0.1, self.τ*y*0.1 ],
                         args = (y),xtol= tol)
            g_k[i] = ans[0]
            g_x[i] = ans[1]

        return g_k,g_x

    def solve_policy_function_principal(self,
                                   solve_tolerance = 1e-9,
                                   max_iter = 50):
        """
        Solves the Model, max_iter defines the max number iterations to be done for coleman operator,
        solve_tolerance is used in fsolve in function self.coleman_operator_principal()
        """
        
        #Generating kstar grid values
        self.kstar_grid_func()
        
        # Initial k and x grids
        k = {}
        x = {}
        k_grid = self.y_grid * ( 1-self.τ ) * 0.5
        x_grid = self.y_grid * (self.τ) * 0.5

        #Initial Plot
        fig, ax = plt.subplots(ncols = 2,figsize=(10, 6))
        lb = 'initial condition$'
        ax[0].plot(self.y_grid, k_grid, color = 'red', lw=2, alpha=0.6, label=lb);
        ax[1].plot(self.y_grid, x_grid, color = 'red', lw=2, alpha=0.6, label=lb);
        
        #Looping Paramters
        j = 0
      
        
        #Looping and plotting
        while j < max_iter:
            new_k_grid, new_x_grid = self.coleman_operator_principal(
                                                                k_grid, 
                                                                x_grid, 
                                                                tol = solve_tolerance
                                                                )
            k_grid[:], x_grid[:] = new_k_grid, new_x_grid
            ax[0].plot(self.y_grid, new_k_grid, color=plt.cm.jet(j / max_iter), lw=2, alpha=0.6);
            ax[1].plot(self.y_grid, new_x_grid, color=plt.cm.jet(j / max_iter), lw=2, alpha=0.6);
            j = j + 1
        
        #Last Policy Plot
        lb = 'Last policy iterations {}'.format(j)
        ax[0].plot(self.y_grid, k_grid, 'black', lw=2, alpha=0.8, label=lb);
        ax[1].plot(self.y_grid, x_grid, color = 'black', lw=2, alpha=0.6, label=lb);

        ### Plots Setup
        
        # Axis Limits
        
        # y limits
        ax[0].set_ylim(0,max(k_grid)+1)
        ax[1].set_ylim(0,max(x_grid)+1)
        
        # x limits
        ax[0].set_xlim(0,max(self.y_grid)+1)
        ax[1].set_xlim(0,max(self.y_grid)+1)
        
        ## labels
        # x label
        ax[0].set_xlabel('Y')
        ax[1].set_xlabel('Y')
        
        # y label
        ax[0].set_ylabel('K')
        ax[1].set_ylabel('X')
        
        # Titles
        ax[0].set_title('K Policy Iterations')
        ax[1].set_title('X Policy Iterations')
        
        # Legends
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='upper left')

        plt.show();
        
        ####  Storing the Final Values in Class
        self.k_grid = k_grid
        self.x_grid = x_grid
        
        ## Creating Value Functions and Plots
        self.create_value_functions_and_plots()
        
    def kstar(self,y):
        """
        Returns optimal value of k in for v*(y)
        """
        return interp1d(self.y_grid, self.kstar_grid,fill_value = 'extrapolate')(y)
        
    def vstar_value(self, y, tol = 1e-5):
        """
        Returns v*(y): Agent value without government contract
        """
        v = {}
        i = 0
        v[i] = 0
        increment = tol + 1
        while increment > tol:
            periodic_utility = self.β**i * np.log(y - self.kstar(y))
            v[i+1] = v[i] + periodic_utility
            increment = np.abs( v[i+1]-v[i] )
            y = self.G(y, 0, self.kstar(y))
            i = i + 1
        return v[i]
    
    def vstar_value_function(self):
        """
        Returns the v* value function
        """
        v = np.zeros_like(self.y_grid)
        for i,y in enumerate(self.y_grid):
            v[i] = self.vstar_value(y)
        self.vstar_value_grid = v

    def k(self,y):
        """
        Returns optimal k for given value of y with government contract
        """
        return interp1d(self.y_grid, self.k_grid,fill_value = 'extrapolate')(y)
    
    def x(self,y):
        """
        Returns optimal k for given value of y with government contract
        """
        return interp1d(self.y_grid,self.x_grid, fill_value = 'extrapolate')(y)
    
    def v_and_w_value(self,y,tol = 1e-5):
        """
        Returns agent present discounted utility and principal present discounted utility for 
        a given value of y with contract
        
        """
        v = {}
        w = {}
        i = 0
        v[i] = 0
        w[i] = 0
        increment = tol + 1
        while increment > tol:
            periodic_agent_utility = self.β**i * np.log( (1-self.τ)*y - self.k(y) )
            periodic_principal_utility = self.β**i * np.log( self.τ*y - self.x(y) )
            v[i+1] = v[i] + periodic_agent_utility
            w[i+1] = w[i] + periodic_principal_utility
            increment = max( np.abs(v[i+1]-v[i]), np.abs(w[i+1]-w[i]) )
            y_next = self.G(y,self.x(y),self.k(y))
            i = i + 1 
        return v[i], w[i]
    
    def v_and_w_value_function(self):
        """
        Returns agent and principal value function with contract
        """
        v = np.zeros_like(self.y_grid)
        w = np.zeros_like(self.y_grid)
        for i,y in enumerate(self.y_grid):
            v[i], w[i] = self.v_and_w_value(y)
        self.v_value_grid = v
        self.w_value_grid = w
    
    def create_value_functions_and_plots(self):
        """
        Creates the value function with and without contract
        """
        
        self.vstar_value_function()
        self.v_and_w_value_function()
        
        fig, ax = plt.subplots(figsize = (10,6));
        ax.plot(self.y_grid, self.vstar_value_grid, 'black', lw=2, alpha=0.8, 
                   label='Agent Value Function without Government contract');
        ax.plot(self.y_grid, self.v_value_grid, 'red', lw=2, alpha=0.8, 
                   label='Agent Value Function with Government Contract');
        ax.set_ylim(0, max ( np.max(self.vstar_value_grid),np.max(self.v_value_grid) ) );
        ax.set_xlim(0,self.y_max);
        ax.set_ylabel('Value')
        ax.set_xlabel('Y ')
        ax.set_title('Agent Value Function')
        ax.legend(loc='lower right');
        plt.show;
        
        fig, ax = plt.subplots(figsize = (10,6));
        ax.plot(self.y_grid, self.w_value_grid, 'blue', lw=2, alpha=0.8, 
                   label='Principal Value Function');
        ax.set_ylim(0,np.max(self.w_value_grid));
        ax.set_xlim(0,self.y_max);
        ax.set_ylabel('Value')
        ax.set_xlabel('Y ')
        ax.set_title('Principal Value Function')
        ax.legend(loc='lower right');
        plt.show;
        
EIPM = Euler_Infinite_Period_Deterministic_Principal_Model(y_max= 200,y_grid_size= 50)
EIPM.solve_policy_function_principal(max_iter = 50)

EIPM.vstar_value(100)
EIPM.v_and_w_value(100)

#####################################################

# 3. Answers

# No Government Contract:

# Given 𝑦0=100, the agent utility is 𝑣∗=39.55

# With Government Contract:

# Given 𝑦0=100, Infinite Period Horizon Model based on Euler Equation yields 𝑣∗=36.87 and 𝑤=28.399.

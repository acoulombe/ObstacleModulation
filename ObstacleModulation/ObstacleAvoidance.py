from .Obstacle import Obstacle
import numpy as np
import matplotlib.pyplot as plt

class ObstacleAvoidance():

    def __init__(self, tail_effects=True) -> None:
        """Initializes Obstacle Avoidance

        Parameters
        ----------
            tail_effects : bool
                boolean that removes (False) or keeps (True) the effects of the obstacle
                modulation after the system passes the obstacle.
        """
        self.tail_effects = tail_effects
        self.obstacle_list = []
        self.dynamics = None

    def add_obstacle(self, obs) -> None:
        """Adds the given obstacle to the environment model for modulation

        Parameters
        ----------
            obs : ObstacleModulation.Obstacle.Obstacle
                Obstacle to add
        
        Raises
        ------
            TypeError
                When provided argument is not of type Obstacle or does not inherit from Obstacle
        """
        if isinstance(obs, Obstacle):
            self.obstacle_list.append(obs)
        else:
            raise TypeError(f"Provided obstacle is of type {type(obs)}, which does not inherit <class ObstacleModulation.Obstacle.Obstacle>")

    def set_system_dynamics(self, dyn) -> None:
        """Set the system dynamics

        Parameters
        ----------
            dyn : callable
                dynamics as a function with state/position as acceptable argument
        """
        self.dynamics = dyn

    def check_collision(self, pos) -> np.array:
        """Queries the  given positions to verify if they are in collision

        Parameters
        ----------
        pos : np.array(N, D)
            2D array where N is the number of positions to query, D is the dimension of the position

        Returns
        ---------
        np.array(N,)
            boolean value of the queried positions (true = in collision, false = no collision)
        """
        in_collision = np.zeros(pos.shape[0], dtype=bool)
        for obs in self.obstacle_list:
            in_collision |= obs.check_collision(pos)

        return in_collision

    def get_gamma(self, pos) -> np.array:
        """
        """
        gamma = np.full(pos.shape[0], float("inf"))
        for obs in self.obstacle_list:
            gamma = np.minimum(gamma, obs.gamma_func(pos))

        return gamma

    def get_action(self, pos, vel=None, tail_effects=False) -> np.array:
        """Gets the velocity for the system based on the current state of the system

        Parameters
        ----------
            pos : np.array(N, D)
                current system position
            vel : np.array(N, D) or None
                current system velocity (if None, the modulation tail effects will always be active)
            tail_effects : bool
                whether to remove the tail effects produces by modulation

        Raises
        ------
            ValueError : exception
                Thrown when the robot motion for the state is not provided either from the velocity argument or the system
                dynamics through `set_system_dynamics`.
        
        Returns
        -------
            np.array
                desired system velocity that follows the provided dynamics while avoiding obstacles
        """
        # Get all obstacle gammas for modulation weighting
        obstacle_gammas = []
        for obs in self.obstacle_list:
            obstacle_gammas.append(obs.gamma_func(pos))

        # product_weights = []
        # for k in range(len(obstacle_gammas)):
        #     weight = 1
        #     for i in range(len(obstacle_gammas)):
        #         if i != k:
        #             weight = weight * (obstacle_gammas[i] - 1)
        #     product_weights.append(weight)
        
        # weights = []
        # denom = np.sum(product_weights, axis=0)
        # for k in range(len(obstacle_gammas)):
        #     w = product_weights[k] / denom
        #     weights.append(w)

        weights = []
        for k in range(len(obstacle_gammas)):
            w = 1
            for i in range(len(obstacle_gammas)):
                if i != k:
                    w = w * (obstacle_gammas[i] - 1) / ((obstacle_gammas[k] - 1) + (obstacle_gammas[i] - 1))
            weights.append(w)
        
        # Get Modulation of each obstacle
        modulations = []
        for i in range(len(self.obstacle_list)):
            obs = self.obstacle_list[i]
            modulations.append(obs.get_modulation_matrix(pos, vel=vel, weight=weights[i], tail_effects=tail_effects))

        M = np.eye(pos.shape[-1])
        for mod in modulations:
            M = np.matmul(M, mod)
        
        # Modulate system
        if vel is not None:
            dyn = vel.reshape(-1, pos.shape[-1], 1)
        elif self.dynamics is not None:
            dyn = self.dynamics(pos).reshape(-1, pos.shape[-1], 1)
        else:
            raise ValueError(f"Robot Motion is undefined:\nProvided velocity {vel}\nProvided system dynamics {self.dynamics}")
        
        mod_dyn = np.matmul(M, dyn).reshape(-1, pos.shape[-1])
        is_zero = np.where(np.all(np.isclose(mod_dyn, 0), axis=1))
        flag = np.any(np.logical_or(np.all(np.isclose(mod_dyn, 0), axis=1), np.isclose(obstacle_gammas, 1))) 
        if flag:
            dyn[is_zero] = np.random.random(dyn.shape)[is_zero]
            mod_dyn = np.matmul(M, dyn).reshape(-1, pos.shape[-1])

        return mod_dyn

    def plot_environment(self, ax, color=['g'], show=False) -> None:
        """Plot all obstacles of the modeled environment

        Parameters
        ----------
            ax : pyplot.axis.Axis
                axis to plot the obstacle on
            color : str
                color to plot the obstacle in
            show : bool
                whether to show the obstacle during the function call
        """
        for i in range(len(self.obstacle_list)):
            self.obstacle_list[i].plot_obstacle(ax, color[i%len(color)])
        
        if show:
            plt.show()
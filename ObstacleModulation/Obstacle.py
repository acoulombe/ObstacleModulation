from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Obstacle(ABC):

    def __init__(self, reference_point, safety_factor=1, reactivity=1, repulsion_coeff=1) -> None:
        """Initializes Obstacle

        Parameters
        ----------
            reference_point : np.array(D,)
                reference position of the obstacle
            safety_factor : np.array(D,) or float
                scalar factor(s) that increases the size of the boundary of the obstacle
                Must be in range [1, \u221E)
            reactivity : float
                scalar factor that determines the influence of the modulation, higher values have higher reactions to obstacle
                and lower values decrease the effect of the obstacle. Value of 1 has basic influence.
                Must be > 0
            repulsion_coeff : float
                scalar factor that determines the repulsion the obstacle puts on the system. Only applied if tail effects are
                removed when modulating the system dynamics
                Must be >= 1
        """
        super().__init__()
        self.ref_pos = reference_point
        self.safety_factor = safety_factor
        self.reactivity = reactivity
        self.repulsion_coeff = repulsion_coeff

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
        return self.gamma_func(pos) < 1

    @abstractmethod
    def sdf(self, pos) -> np.array:
        """Signed Distance function indicating the distance between the robot and the obstacle boundary, where < 0
        represents the penetration distance and > 0 represents the separation distance and 0 is touching the boundary

        Parameters
        ----------
        pos : np.array(N, D)
            2D array where N is the number of positions to query, D is the dimension of the position

        Returns
        ---------
        np.array(N,)
            SDF values of the queried positions
        """
        pass

    @abstractmethod
    def gamma_func(self, pos) -> np.array:
        """Gamma function indicating the ratio of the robot to the obstacle boundary,
        where 1 represents the obstacle boundary, > 1 is outside the obstacle and < 1 is inside the obstacle

        Parameters
        ----------
        pos : np.array(N, D)
            2D array where N is the number of positions to query, D is the dimension of the position

        Returns
        ---------
        np.array(N,)
            Gamma values of the queried positions
        """
        pass

    def get_eigenvalues(self, pos, weight=1, tail_effects=False) -> Tuple[np.array, np.array]:
        """Gets the eigenvalues of the modulation matrix for the given position of the system

        Parameters
        ----------
        pos : np.array(N, D)
            2D array where N is the number of positions to query, D is the dimension of the position
        weight : float
            weight of the obstacle when applying modulation on the system
        tail_effects : bool
                whether to remove the tail effect of the modulation when moving away from the obstacle

        Returns
        ---------
        np.array(N,)
            eigenvalue for the normal/reference basis vector
        np.array(N,)
            eigenvalue for the tangential basis vectors
        """
        gamma = self.gamma_func(pos)
        if tail_effects is False:
            react_gamma = (gamma  ** (1/self.reactivity))
        else:
            react_gamma = (gamma/self.repulsion_coeff  ** (1/self.reactivity))

        lambda_r = 1 - weight / react_gamma
        lambda_r[tail_effects] = 1

        lambda_e = 1 + weight / react_gamma

        return lambda_r, lambda_e

    def get_diagonal_matrix(self, pos, weight=1, tail_effects=False) -> np.array:
        """Gets the diagonal matrices to scale the modulation of the system around the obstacle for
        all given positions

        Parameters
        ----------
        pos : np.array(N, D)
            2D array where N is the number of positions to query, D is the dimension of the position
        weight : float
            weight of the obstacle when applying modulation on the system
        tail_effects : bool
            whether to remove the tail effect of the modulation when moving away from the obstacle

        Returns
        ---------
        np.array(N, D, D)
            diagonal matrices from the eigenvalues taken from the distance of the system to the obstacle
        """
        lambda_r, lambda_e = self.get_eigenvalues(pos, weight=weight, tail_effects=tail_effects)
        D = np.zeros((pos.shape[0], pos.shape[1], pos.shape[1]))
        for idx in range(pos.shape[1]):
            if idx == 0:
                D[:, idx, idx] = lambda_r
            else:
                D[:, idx, idx] = lambda_e

        return D

    @abstractmethod
    def get_basis_matrix(self, pos) -> np.array:
        """Gets the basis matrices to form the vector space for modulating the system around the obstacle for
        all given positions

        Parameters
        ----------
        pos : np.array(N, D)
            2D array where N is the number of positions to query, D is the dimension of the position

        Returns
        ---------
        np.array(N, D, D)
            basis matrices from the eigenvectors formed by the object normal/reference vector and tangential
            vectors to the object boundary
        """
        pass

    def get_modulation_matrix(self, pos, vel=None, weight=1, tail_effects=False) -> np.array:
        """Gets the modulation matrix to modulating the system around the obstacle for
        all given positions

        Parameters
        ----------
        pos : np.array(N, D)
            2D array where N is the number of positions to query, D is the dimension of the position
        weight : float
            weight of the obstacle when applying modulation on the system
        tail_effects : bool
            whether to remove the tail effect of the modulation when moving away from the obstacle

        Returns
        ---------
        np.array(N, D, D)
            modulation matrices to stretch the space around the obstacle to move the system around the boundary
        """
        E = self.get_basis_matrix(pos)
        if vel is not None and tail_effects is True:
            tail_effects = (np.sum(E[:,:,0] * vel, axis=1) >= 0)
        D = self.get_diagonal_matrix(pos, weight=weight, tail_effects=tail_effects)
        invE = np.linalg.pinv(E)
        M = np.matmul(np.matmul(E, D), invE)
        return M

    @abstractmethod
    def plot_obstacle(self, ax, color='g', show=False) -> None:
        """Plot the obstacle

        Parameters
        ----------
            ax : pyplot.axis.Axis
                axis to plot the obstacle on
            color : str
                color to plot the obstacle in
            show : bool
                whether to show the obstacle during the function call
        """
        pass

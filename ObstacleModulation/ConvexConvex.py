from typing import Tuple

import Collision
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull as ConvHull

from .Obstacle import Obstacle


class ConvexConvex(Obstacle):

    def __init__(self, reference_point, vertices, orientation, safety_factor=1, reactivity=1, repulsion_coeff=1):
        """Initializes Cuboid Obstacle

        Parameters
        ----------
            reference_point : np.array(D,)
                reference position of the obstacle
            vertices : np.array(N, D)
                vertices describing the shape of the convex hull
            orientation : np.array(D, D)
                Rotation matrix describing the orientation of the cuboid
            safety_factor : float
                scalar factor that increases the size of the boundary of the obstacle by representing the object
                Must be in range [1, \u221E)
            reactivity : float
                scalar factor that determines the influence of the modulation, higher values have higher reactions to obstacle
                and lower values decrease the effect of the obstacle. Value of 1 has basic influence.
                Must be > 0
            repulsion_coeff : float
                scalar factor that determines the repulsion the obstacle puts on the system
                Must be >= 1
        """
        super().__init__(reference_point, safety_factor=safety_factor, reactivity=reactivity, repulsion_coeff=repulsion_coeff)
        self.orientation = orientation
        self.vertices = vertices

    def check_collision(self, body_ref, body) -> np.array:
        """Queries the  given positions to verify if they are in collision

        Parameters
        ----------
        body_ref : np.array(D,)
            reference position of the queried body
        body : np.array(M, D)
            2D array where M is the number of vertices in the body, D is the dimension of space

        Returns
        ---------
        np.array(N,)
            boolean value of the queried positions (true = in collision, false = no collision)
        """
        return self.gamma_func(body_ref, body) < 1

    def sdf(self, body_ref, body)  -> np.array:
        """Signed Distance function indicating the distance of the robot body to the obstacle boundary,
        where 0 represents the obstacle boundary, > 0 is the separating distance with the obstacle and
        < 0 is the penetration distance with the obstacle

        Parameters
        ----------
        body_ref : np.array(D,)
            reference position of the queried body
        body : np.array(M, D)
            2D array where M is the number of vertices in the body, D is the dimension of space

        Returns
        ---------
        float
            SDF value
        """
        body_in_local_rf = np.matmul(self.orientation, (body - self.ref_pos).T).T

        # Get Face point closest to body
        if(self.ref_pos.shape[0] == 2):
            sd, D, cache = Collision.signedDistance2d(body_in_local_rf, self.vertices * self.safety_factor)
        elif(self.ref_pos.shape[0] == 3):
            sd, D, cache = Collision.signedDistance(body_in_local_rf, self.vertices * self.safety_factor)
        else:
            raise NotImplementedError("Only Dimensions 2 and 3 are supported")

        return sd

    def gamma_func(self, body_ref, body)  -> np.array:
        """Gamma function indicating the ratio of the robot body to the obstacle boundary,
        where 1 represents the obstacle boundary, > 1 is outside the obstacle and < 1 is inside the obstacle

        Parameters
        ----------
        body_ref : np.array(D,)
            reference position of the queried body
        body : np.array(M, D)
            2D array where M is the number of vertices in the body, D is the dimension of space

        Returns
        ---------
        float
            Gamma value
        """
        body_in_local_rf = np.matmul(self.orientation, (body - self.ref_pos).T).T

        # Get Face point closest to body
        if(self.ref_pos.shape[0] == 2):
            sd, D, cache = Collision.signedDistance2d(body_in_local_rf, self.vertices * self.safety_factor)
        elif(self.ref_pos.shape[0] == 3):
            sd, D, cache = Collision.signedDistance(body_in_local_rf, self.vertices * self.safety_factor)
        else:
            raise NotImplementedError("Only Dimensions 2 and 3 are supported")

        gamma = (np.linalg.norm(body_ref - self.ref_pos)) / (np.linalg.norm(body_ref -  self.ref_pos) - sd)

        return gamma

    def get_eigenvalues(self, body_ref, body, weight=1, tail_effects=False) -> Tuple[np.array, np.array]:
        """Gets the eigenvalues of the modulation matrix for the given body position of the system

        Parameters
        ----------
        body_ref : np.array(D,)
            reference position of the queried body
        body : np.array(M, D)
            2D array where M is the number of vertices in the body, D is the dimension of space
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
        gamma = self.gamma_func(body_ref, body)
        if tail_effects is False:
            react_gamma = (gamma  ** (1/self.reactivity))
        else:
            react_gamma = (gamma/self.repulsion_coeff  ** (1/self.reactivity))

        lambda_r = 1 - weight / react_gamma
        if tail_effects:
            lambda_r = 1

        lambda_e = 1 + weight / react_gamma

        return lambda_r, lambda_e

    def get_diagonal_matrix(self, body_ref, body, weight=1, tail_effects=False) -> np.array:
        """Gets the diagonal matrix to scale the modulation of the system around the obstacle for
        all given body position

        Parameters
        ----------
        body_ref : np.array(D,)
            reference position of the queried body
        body : np.array(M, D)
            2D array where M is the number of vertices in the body, D is the dimension of space
        weight : float
            weight of the obstacle when applying modulation on the system
        tail_effects : bool
            whether to remove the tail effect of the modulation when moving away from the obstacle

        Returns
        ---------
        np.array(D, D)
            diagonal matrix from the eigenvalues taken from the distance of the system to the obstacle
        """
        lambda_r, lambda_e = self.get_eigenvalues(body_ref, body, weight=weight, tail_effects=tail_effects)
        D = np.zeros((body.shape[1], body.shape[1]))
        for idx in range(body.shape[1]):
            if idx == 0:
                D[idx, idx] = lambda_r
            else:
                D[idx, idx] = lambda_e

        return D

    def get_basis_matrix(self, body_ref, body)  -> np.array:
        """Gets the basis matrix to form the vector space for modulating the system around the obstacle for
        all given body

        Parameters
        ----------
        body_ref : np.array(D,)
            reference position of the queried body
        body : np.array(M, D)
            2D array where M is the number of vertices in the body, D is the dimension of space

        Returns
        ---------
        np.array(D,D)
            Basis matrix from the eigenvectors formed by the object normal/reference vector and tangential
            vectors to the object boundary
        """
        E = np.zeros((body.shape[1], body.shape[1]))

        n = (body_ref - self.ref_pos).flatten()
        n_norm = np.linalg.norm(n)
        if not np.isclose(n_norm, 0):
            n = n / n_norm

        # Get tangential vectors of surface
        body_in_local_rf = np.matmul(self.orientation, (body - self.ref_pos).T).T

        if self.ref_pos.shape[0] == 2: # 2-dimension
            sd, D, cache = Collision.signedDistance2d(body_in_local_rf, self.vertices * self.safety_factor)

            def getOrthogonal2d(n):
                e = np.zeros([n.shape[0]-1, n.shape[0]])
                e[0,0] = -n[1]
                e[0,1] = n[0]
                return e

            if np.isclose(cache.A[0] + D, cache.A[1]).all():
                # use circular coordinates to get tangential vector to surface
                e = getOrthogonal2d(n)
            else:
                D = (self.orientation.T).dot(D)
                D_norm = np.linalg.norm(D)
                if not np.isclose(D_norm, 0):
                    D = -D / D_norm
                    e = getOrthogonal2d(D)
                else:
                    e = getOrthogonal2d(n)

        elif self.ref_pos.shape[0] == 3: # 3-dimension
            sd, D, cache = Collision.signedDistance(body_in_local_rf, self.vertices * self.safety_factor)

            def getOrthogonal3d(n):
                e = np.zeros([n.shape[0]-1, n.shape[0]])
                # use spherical coordinates to get tangential vectors to surface
                cosTheta = n[2]
                theta = np.arccos(cosTheta)
                sinTheta = np.sin(theta)
                if np.isclose(sinTheta, 0):
                    cosPhi = 1
                    sinPhi = 0
                else:
                    cosPhi = n[0] / sinTheta
                    sinPhi = n[1] / sinTheta

                # Theta unit vector
                e[0,0] = cosTheta * cosPhi
                e[0,1] = cosTheta * sinPhi
                e[0,2] = -sinTheta

                # Phi unit vector
                e[1,0] = -sinPhi
                e[1,1] = cosPhi
                # last unit vector value is 0
                return e

            if np.isclose(cache.A[0] + D, cache.A[1]).all():
                e = getOrthogonal3d(n)
            else:
                D = (self.orientation.T).dot(D)
                D_norm = np.linalg.norm(D)
                if not np.isclose(D_norm, 0):
                    D = -D / D_norm
                    e = getOrthogonal3d(D)
                else:
                    e = getOrthogonal3d(n)

        E[0,:] = n
        E[1:,:] = e
        E = E.T

        if np.linalg.det(E) == 0:
            print("singular")

        return E

    def get_modulation_matrix(self, body_ref, body, vel=None, weight=1, tail_effects=False) -> np.array:
        """Gets the modulation matrix to modulating the system around the obstacle for
        all given body position

        Parameters
        ----------
        body_ref : np.array(D,)
            reference position of the queried body
        body : np.array(M, D)
            2D array where M is the number of vertices in the body, D is the dimension of space
        weight : float
            weight of the obstacle when applying modulation on the system
        tail_effects : bool
            whether to remove the tail effect of the modulation when moving away from the obstacle

        Returns
        ---------
        np.array(D, D)
            modulation matrix to stretch the space around the obstacle to move the system around the boundary
        """
        E = self.get_basis_matrix(body_ref, body)
        if vel is not None and tail_effects is True:
            tail_effects = (np.sum(E[:,0] * vel, axis=1) >= 0)
        D = self.get_diagonal_matrix(body_ref, body, weight=weight, tail_effects=tail_effects)
        invE = np.linalg.pinv(E)
        M = np.matmul(np.matmul(E, D), invE)
        return M

    def plot_obstacle(self, ax, color='g', show=False) -> None:
        if self.ref_pos.shape[0] == 2:
            points = (self.orientation.T).dot(self.vertices.T).T + self.ref_pos
            hull = ConvHull(points)
            # draw the polygons of the convex hull
            ax.fill(points[hull.vertices,0], points[hull.vertices,1], color=color)

            outline_points = (self.orientation.T).dot(self.vertices.T * self.safety_factor).T + self.ref_pos
            hull = ConvHull(outline_points)
            # draw the polygons of the convex hull
            ax.fill(outline_points[hull.vertices,0], outline_points[hull.vertices,1], 'k--', alpha=0.2)

        elif self.ref_pos.shape[0] == 3:
            vertices = (self.orientation.T).dot(self.vertices.T).T + self.ref_pos
            hull = ConvHull(vertices)
            # draw the polygons of the convex hull
            for s in hull.simplices:
                v = vertices[s].reshape(1, 3, 3)
                tri = Poly3DCollection(v, facecolor=color)
                ax.add_collection3d(tri)


            # Outline
            vertices = (self.orientation.T).dot(self.vertices.T * self.safety_factor).T + self.ref_pos
            hull = ConvHull(vertices)
            # draw the polygons of the convex hull
            for s in hull.simplices:
                v = vertices[s].reshape(1, 3, 3)
                tri = Poly3DCollection(v, facecolor='gray', linewidth=1, edgecolor='k', alpha=0.2)
                ax.add_collection3d(tri)
        else:
            raise NotImplementedError("Only Dimensions 2 and 3 are supported")


        if show:
            plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull as ConvHull

from .Obstacle import Obstacle
import Collision

class ConvexHull(Obstacle):

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

    def gamma_func(self, pos)  -> np.array:
        pos_in_local_rf = np.matmul(self.orientation, (pos - self.ref_pos).T).T

        # Get Face point closest to body
        queried_dist = []
        for i in range(0, len(pos_in_local_rf)):
            if(self.ref_pos.shape[0] == 2):
                sd, D, cache = Collision.signedDistance2d(pos_in_local_rf[i,:].reshape(-1,2), self.vertices * self.safety_factor)
                P1 = cache.A[0]
                P2 = cache.A[0] + D
            elif(self.ref_pos.shape[0] == 3):
                sd, D, cache = Collision.signedDistance(pos_in_local_rf[i,:].reshape(-1,3), self.vertices * self.safety_factor)
                P1, P2 = Collision.closestPoints(D, cache)
            else:
                raise NotImplementedError("Only Dimensions 2 and 3 are supported")
            
            dist = (sd + np.linalg.norm(P2)) / np.linalg.norm(P2)
            queried_dist.append(dist)
    
        # Get Gamma values
        gammas = np.array(queried_dist)

        return gammas

    def get_basis_matrix(self, pos)  -> np.array:
        E = np.zeros((pos.shape[0], pos.shape[1], pos.shape[1]))

        n = pos - self.ref_pos
        n_norm = np.linalg.norm(n, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            n = n / n_norm.reshape(-1,1)
            n[np.isclose(n_norm, 0)] = 0

        # Get tangential vectors of surface
        pos_in_local_rf = np.matmul(self.orientation, (pos - self.ref_pos).T).T

        for i in range(0, len(pos_in_local_rf)):
            # e = np.zeros([self.ref_pos.shape[0]-1, self.ref_pos.shape[0]])
            if self.ref_pos.shape[0] == 2: # 2-dimension
                sd, D, cache = Collision.signedDistance2d(pos_in_local_rf[i,:].reshape(-1,2), self.vertices * self.safety_factor)

                def getOrthogonal2d(n):
                    e = np.zeros([n.shape[0]-1, n.shape[0]])
                    e[0,0] = -n[1]
                    e[0,1] = n[0]
                    return e

                if np.isclose(cache.A[0] + D, cache.A[1]).all():
                    # use circular coordinates to get tangential vector to surface
                    e = getOrthogonal2d(n[i])
                else:
                    D = (self.orientation.T).dot(D)
                    D_norm = np.linalg.norm(D)
                    if not np.isclose(D_norm, 0):
                        D = -D / D_norm
                        e = getOrthogonal2d(D)
                    else:
                        e = getOrthogonal2d(n[i])

            elif self.ref_pos.shape[0] == 3: # 3-dimension
                sd, D, cache = Collision.signedDistance(pos_in_local_rf[i,:].reshape(-1,3), self.vertices * self.safety_factor)

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
                    e = getOrthogonal3d(n[i])
                else:
                    D = (self.orientation.T).dot(D)
                    D_norm = np.linalg.norm(D)
                    if not np.isclose(D_norm, 0):
                        D = -D / D_norm
                        e = getOrthogonal3d(D)
                    else:
                        e = getOrthogonal3d(n[i])
            
            E[i,0] = n[i,:]
            E[i,1:] = e
            E[i] = E[i].T

            if np.linalg.det(E[i]) == 0:
                print("singular")

        return E

    def plot_obstacle(self, ax, color='g', show=False) -> None:
        if self.ref_pos.shape[0] == 2:
            points = (self.orientation.T).dot(self.vertices.T).T + self.ref_pos
            rectangle = Polygon(points, color=color)
            
            outline_points = (self.orientation.T).dot(self.vertices.T * self.safety_factor).T + self.ref_pos
            outline = Polygon(outline_points, color='k', fill=False, linestyle='--')
            
            ax.add_patch(rectangle)
            ax.add_patch(outline)

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

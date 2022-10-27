import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

from .Obstacle import Obstacle


class SphereInternal(Obstacle):

    def __init__(self, reference_point, radius, safety_factor=1, reactivity=1):
        """Initializes Sphere Obstacle

        Parameters
        ----------
            reference_point : np.array(D,)
                reference position of the obstacle
            radius : float
                Radius of the sphere
            safety_factor : float
                scalar factor that increases the size of the boundary of the obstacle
                Must be in range [1, \u221E)
            reactivity : float
                scalar factor that determines the influence of the modulation, higher values have higher reactions to obstacle
                and lower values decrease the effect of the obstacle. Value of 1 has basic influence.
                Must be > 0
        """
        super().__init__(reference_point, safety_factor=safety_factor, reactivity=reactivity)
        self.radius = radius

    def sdf(self, pos) -> np.array:
        return (np.linalg.norm(pos - self.ref_pos, axis=1) - self.radius) * self.safety_factor

    def gamma_func(self, pos)  -> np.array:
        dist = np.linalg.norm(pos - self.ref_pos, axis=1) / self.safety_factor
        return self.radius / dist

    def get_basis_matrix(self, pos)  -> np.array:
        E = np.zeros((pos.shape[0], pos.shape[1], pos.shape[1]))

        r = pos - self.ref_pos
        r_norm = np.linalg.norm(r, axis=1)
        mask = ~np.isclose(r_norm, 0)
        n = np.zeros(r.shape)
        n[mask] = r[mask] / r_norm[mask].reshape(-1,1)
        n[~mask] = [1,0]

        # Get tangential vectors of surface
        e = np.zeros((pos.shape[0], pos.shape[1], self.ref_pos.shape[0]-1))
        if self.ref_pos.shape[0] == 2: # 2 dimensional aka circle
            # use circular coordinates to get tangential vector to surface
            e[:,0,0] = -n[:,1]
            e[:,1,0] = n[:,0]

        elif self.ref_pos.shape[0] == 3: # 3 dimensional aka sphere
            # use spherical coordinates to get tangential vectors to surface
            cosTheta = n[:,2]
            theta = np.arccos(cosTheta)
            sinTheta = np.sin(theta)
            cosPhi = n[:,0] / sinTheta
            sinPhi = n[:,1] / sinTheta

            # Theta unit vector
            e[:,0,0] = cosTheta * cosPhi
            e[:,1,0] = cosTheta * sinPhi
            e[:,2,0] = -sinTheta

            # Phi unit vector
            e[:,0,1] = -sinPhi
            e[:,1,1] = cosPhi
            # last unit vector value is 0

        else:
            raise NotImplementedError(f"Only defined for 2 and 3 dimensions, which are circle and sphere. Dimension provided {self.ref_pos.shape[0]}")

        E[:,:,0] = n
        E[:,:,1:] = e

        return E

    def plot_obstacle(self, ax, color='g', show=False) -> None:
        if self.ref_pos.shape[0] == 2:
            circle = plt.Circle(self.ref_pos, self.radius, color=color, fill=False)
            outline = plt.Circle(self.ref_pos, self.radius / self.safety_factor, color='k', fill=False, linestyle='--')
            ax.add_patch(circle)
            ax.add_patch(outline)
        elif self.ref_pos.shape[0] == 3:
            N = 100
            u = np.linspace(0, 2 * np.pi, N)
            v = np.linspace(0, np.pi, N)
            # Sphere
            x = np.outer(np.cos(u), np.sin(v)) * self.radius + self.ref_pos[0]
            y = np.outer(np.sin(u), np.sin(v)) * self.radius + self.ref_pos[1]
            z = np.outer(np.ones(np.size(u)), np.cos(v)) * self.radius + self.ref_pos[2]

            ax.plot_surface(x, y, z, linewidth=0.0, color=color, fill=False)

            # Safety region
            x = np.outer(np.cos(u), np.sin(v)) * self.radius / self.safety_factor + self.ref_pos[0]
            y = np.outer(np.sin(u), np.sin(v)) * self.radius / self.safety_factor + self.ref_pos[1]
            z = np.outer(np.ones(np.size(u)), np.cos(v)) * self.radius * self.safety_factor + self.ref_pos[2]

            ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='k', alpha=0.15)
        else:
            raise NotImplementedError("Only Dimensions 2 and 3 are supported")

        if show:
            plt.show()

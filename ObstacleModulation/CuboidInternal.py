import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .Obstacle import Obstacle

class CuboidInternal(Obstacle):

    def __init__(self, reference_point, orientation, dimensions, safety_factor=1, reactivity=1):
        """Initializes Cuboid Obstacle

        Parameters
        ----------
            reference_point : np.array(D,)
                reference position of the obstacle
            orientation : np.array(D, D)
                Rotation matrix describing the orientation of the cuboid
            dimensions : np.array(D,)
                dimensions of the volume of the cuboid as (x_length, y_length, z_length)
            safety_factor : np.array(D,)
                scalar factor that increases the size of the boundary of the obstacle by representing the object
                Must be in range [1, \u221E)
            reactivity : float
                scalar factor that determines the influence of the modulation, higher values have higher reactions to obstacle
                and lower values decrease the effect of the obstacle. Value of 1 has basic influence.
                Must be > 0
        """
        super().__init__(reference_point, safety_factor=safety_factor, reactivity=reactivity)
        self.orientation = orientation
        self.dimensions = dimensions
        self.halfDim = dimensions / 2
        self.shellDim = self.halfDim / self.safety_factor

    def gamma_func(self, pos)  -> np.array:
        pos_in_local_rf = np.matmul(self.orientation, (pos - self.ref_pos).T).T

        q = np.abs(pos_in_local_rf) - self.shellDim
        q_pos = np.copy(q)
        q_pos[q_pos < 0] = 0
        dist = np.linalg.norm(q_pos, axis=1) + np.minimum(np.max(q, axis=1), 0)

        bound_dist = np.tile(self.halfDim, (q.shape[0], 1))
        boundary = np.copy(bound_dist)
        boundary[q < 0] = 0
        
        # Fix internal face distances
        tmp = np.abs(pos_in_local_rf)[np.all(boundary == 0, axis=1)]
        selection = q[np.all(boundary == 0, axis=1)]
        mask = np.argwhere(selection == np.amax(selection, axis=1)[:,None])
        tmp[mask[:,0], mask[:,1]] = bound_dist[mask[:,0], mask[:,1]]
        boundary[np.all(boundary == 0, axis=1)] = tmp

        # Fix outer face distance
        boundary[boundary == 0] = np.abs(pos_in_local_rf)[boundary == 0]

        face_dist = np.linalg.norm(boundary, axis=1)

        gammas = (dist + face_dist) / face_dist

        return 1 / gammas

    def get_basis_matrix(self, pos)  -> np.array:
        E = np.zeros((pos.shape[0], pos.shape[1], pos.shape[1]))

        n = pos - self.ref_pos
        n_norm = np.linalg.norm(n, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            n = n / n_norm.reshape(-1,1)
            n[np.isclose(n_norm, 0)] = [1,0]

        # Get tangential vectors of surface
        pos_in_local_rf = np.matmul(self.orientation, (pos - self.ref_pos).T).T

        # Get face the point is projected onto
        face_agreement = np.abs(pos_in_local_rf) / self.shellDim
        face = np.argmax(face_agreement, axis=1)
    
        eigenvectors = np.tile(self.orientation, (pos.shape[0], 1, 1))
        mask = np.zeros_like(eigenvectors, bool)
        mask[np.arange(pos.shape[0]), face, :] = True
        e = np.delete(eigenvectors, mask.flatten()).reshape(-1, self.ref_pos.shape[0]-1, self.ref_pos.shape[0])

        E[:,:,0] = n
        E[:,:,1:] = np.transpose(e, (0, 2, 1))

        return E

    def plot_obstacle(self, ax, color='g', show=False) -> None:
        if self.ref_pos.shape[0] == 2:
            anchor = self.ref_pos - (self.orientation.T).dot(self.dimensions)/2
            rectangle = Rectangle(anchor, self.dimensions[0], self.dimensions[1], 180 / np.pi * np.arctan2(self.orientation[0,1], self.orientation[0,0]), color=color, fill=False)
            
            dim_outline = self.dimensions / self.safety_factor
            anchor_outline = self.ref_pos - (self.orientation.T).dot(dim_outline)/2
            outline = Rectangle(anchor_outline, dim_outline[0], dim_outline[1],  180 / np.pi * np.arctan2(self.orientation[0,1], self.orientation[0,0]), color='k', fill=False, linestyle='--')
            
            ax.add_patch(rectangle)
            ax.add_patch(outline)

        elif self.ref_pos.shape[0] == 3:
            # Physical Cuboid
            max_point = self.halfDim
            min_point = -self.halfDim
            vertices_og = np.array([
                [min_point[0], min_point[1], min_point[2]],
                [min_point[0], min_point[1], max_point[2]],
                [min_point[0], max_point[1], min_point[2]],
                [min_point[0], max_point[1], max_point[2]],
                [max_point[0], min_point[1], min_point[2]],
                [max_point[0], min_point[1], max_point[2]],
                [max_point[0], max_point[1], min_point[2]],
                [max_point[0], max_point[1], max_point[2]],
            ])

            vertices = (self.orientation.T).dot(vertices_og.T).T + self.ref_pos

            faces = np.array([
                [vertices[0, :], vertices[1, :], vertices[3, :], vertices[2, :]],
                [vertices[4, :], vertices[5, :], vertices[7, :], vertices[6, :]],
                [vertices[0, :], vertices[1, :], vertices[5, :], vertices[4, :]],
                [vertices[2, :], vertices[3, :], vertices[7, :], vertices[6, :]],
                [vertices[0, :], vertices[2, :], vertices[6, :], vertices[4, :]],
                [vertices[1, :], vertices[3, :], vertices[7, :], vertices[5, :]],
            ])

            prism = Poly3DCollection(faces, facecolor=color)

            ax.add_collection3d(prism)


            # Outline
            max_point = self.shellDim
            min_point = -self.shellDim

            vertices_og = np.array([
                [min_point[0], min_point[1], min_point[2]],
                [min_point[0], min_point[1], max_point[2]],
                [min_point[0], max_point[1], min_point[2]],
                [min_point[0], max_point[1], max_point[2]],
                [max_point[0], min_point[1], min_point[2]],
                [max_point[0], min_point[1], max_point[2]],
                [max_point[0], max_point[1], min_point[2]],
                [max_point[0], max_point[1], max_point[2]],
            ])

            vertices = (self.orientation.T).dot(vertices_og.T).T + self.ref_pos

            faces = np.array([
                [vertices[0, :], vertices[1, :], vertices[3, :], vertices[2, :]],
                [vertices[4, :], vertices[5, :], vertices[7, :], vertices[6, :]],
                [vertices[0, :], vertices[1, :], vertices[5, :], vertices[4, :]],
                [vertices[2, :], vertices[3, :], vertices[7, :], vertices[6, :]],
                [vertices[0, :], vertices[2, :], vertices[6, :], vertices[4, :]],
                [vertices[1, :], vertices[3, :], vertices[7, :], vertices[5, :]],
            ])

            prism = Poly3DCollection(faces, facecolor='gray', linewidth=1, edgecolor='k', alpha=0.2)

            ax.add_collection3d(prism)
        else:
            raise NotImplementedError("Only Dimensions 2 and 3 are supported")


        if show:
            plt.show()

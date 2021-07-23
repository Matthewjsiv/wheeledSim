import torch
import numpy as np
import matplotlib.pyplot as plt

def heatmap(x):
    plt.imshow(x)
    plt.show()

def sobel_x(device):
    return torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1, -2., -1.]]).to(device)

def sobel_y(device):
    return torch.tensor([[1., 0., -1.], [2., 0., -2.], [1, 0., -1.]]).to(device)

class HeightMap:
    """
    Map class with a few additional features to make force calculations easier.
    Namely, allow for querying of normal vectors at various points.
    """
    def __init__(self, data, resolution, k_smooth=1.0, k_slope=0.5, k_curvature=0.05):
        """
        Args:
            data: The data to take as the map. Assume that the map is centered at (0, 0).
                Also note that scaling and stuff is weird because images =/= matrices.
                By convention, the first index is along x, and the second is along y.
                This will be reversed if you access the data field directly.

            resolution: The width/height of each grid cell in m
        """
        self.data = data
        self.resolution = resolution
        self.ox = self.data.shape[1]//2
        self.oy = self.data.shape[0]//2
        self.width = 0.5 * self.data.shape[1] * resolution
        self.height = 0.5 * self.data.shape[0] * resolution
        self.precompute_normals()
        self.precompute_cost(k_smooth, k_slope, k_curvature)

    def precompute_normals(self):
        """
        Precompute the normal for each map point.
        Fit a plane to the nearest neighbors and take that. (i.e. normal = smallest vec of svd of your 4-neighbors)
        Also, pad once. The boundary values will be a little off.
        """
        #Some massaging to get torch to recognize as image.
        data_pad = torch.nn.functional.pad(self.data.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze()
        #SVD matrix should be xN x yN x 4 x 3
        X = torch.zeros(self.data.shape[0], self.data.shape[1], 4, 3)
        X[:, :, 0, 0] = self.resolution  #Right
        X[:, :, 0, 2] = data_pad[2:, 1:-1] - self.data

        X[:, :, 1, 0] = -self.resolution #left
        X[:, :, 1, 2] = data_pad[:-2, 1:-1] - self.data

        X[:, :, 2, 1] = self.resolution  #up
        X[:, :, 2, 2] = data_pad[1:-1, 2:] - self.data

        X[:, :, 3, 1] = -self.resolution #down
        X[:, :, 3, 2] = data_pad[1:-1, :-2] - self.data

        U, S, V = torch.linalg.svd(X)

        normals = V[:, :, -1]
        normals *= normals[:, :, -1].sign().unsqueeze(-1)

        self.normals = normals

    def get_height(self, idxs):
        """
        Get the (interpolated) height at (x, y)
        """
        yp, xp = idxs

        if isinstance(xp, torch.Tensor):
            xq = (xp / self.resolution).long() + self.ox
            yq = (yp / self.resolution).long() + self.oy
        else:
            xq = int(xp / self.resolution) + self.ox
            yq = int(yp / self.resolution) + self.oy
        
        return self.data[xq, yq]

    def get_cost(self, idxs):
        """
        Get the (interpolated) cost at (x, y)
        """
        yp, xp = idxs

        if isinstance(xp, torch.Tensor):
            xq = (xp / self.resolution).long() + self.ox
            yq = (yp / self.resolution).long() + self.oy
        else:
            xq = int(xp / self.resolution) + self.ox
            yq = int(yp / self.resolution) + self.oy

        mask = (xq < 0) | (xq >= self.data.shape[1]) | (yq < 0) | (yq >= self.data.shape[0])
        xq2 = xq.clamp(0, self.data.shape[1]-1)
        yq2 = yq.clamp(0, self.data.shape[0]-1)

        cost = self.cost[xq2, yq2]
        cost[mask] = float('inf')

        return cost

    def get_normal(self, idxs):
        """
        Get the (interpolated) normal at (x, y)
        """
        yp, xp = idxs

        if isinstance(xp, torch.Tensor):
            xq = (xp / self.resolution).long() + self.ox
            yq = (yp / self.resolution).long() + self.oy
        else:
            xq = int(xp / self.resolution) + self.ox
            yq = int(yp / self.resolution) + self.oy
        
        return self.normals[xq, yq]

    def precompute_cost(self, k_smooth=1.0, k_curvature=1.0, k_slope=1.0):
        """
        Get a cell-wise cost and clamp it to 0, 1.
        """
        cost = k_smooth * self.get_smoothness() + k_curvature * self.get_curvature() + k_slope * self.get_slope()
        self.cost = cost.clamp(0, 1)

    def get_smoothness(self, kernel_size=3):
        """
        Get smoothness as height stddev in a local neighborhood.
        """
        r = int(kernel_size/2)
        data_pad = torch.nn.functional.pad(self.data.unsqueeze(0).unsqueeze(0), (r, r, r, r), mode='replicate').squeeze()
        #I'm not sure the fast way to do this. I'm just for-looping for now.
        data_raster = torch.zeros(*self.data.shape, kernel_size**2).to(data_pad.device)
        for i in range(kernel_size):
            for j in range(kernel_size):
                data_raster[:, :, i*kernel_size + j] = data_pad[i:i+self.data.shape[0], j:j+self.data.shape[1]]

        return data_raster.std(dim=-1)

    def get_slope(self):
        """
        Get slope as angle bet. normal and gravity. Higher slope = higher cost.
        """
        g = torch.tensor([0., 0., 1.], device=self.normals.device) #Normals face upward by convention.
        angle = (self.normals * g).sum(dim=-1) # e [0, 1]
        return 1. - angle

    def get_curvature(self):
        """
        Get curvature as magnitude of change in normal directions.
        Do this by getting magnitude of Gxx, Gyy
        """
        sx = sobel_x(self.data.device)
        sy = sobel_y(self.data.device)

        gx = torch.nn.functional.conv2d(self.data.unsqueeze(0).unsqueeze(0), weight=sx.unsqueeze(0).unsqueeze(0), padding=1)
        gy = torch.nn.functional.conv2d(self.data.unsqueeze(0).unsqueeze(0), weight=sy.unsqueeze(0).unsqueeze(0), padding=1)
        Gxx = torch.nn.functional.conv2d(gx, weight=sx.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
        Gyy = torch.nn.functional.conv2d(gy, weight=sy.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
        K = torch.hypot(Gxx, Gyy)
        return K

    def render_3d(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        grid = torch.meshgrid(torch.linspace(-self.width, self.width, self.data.shape[0]), torch.linspace(-self.height, self.height, self.data.shape[1]))
        ax.plot_wireframe(grid[0], grid[1], self.data)

        for i in range(1, grid[0].shape[0]-1):
            for j in range(1, grid[0].shape[1]-1):
                x = grid[0][i, j]
                y = grid[1][i, j]
                z = self.data[i, j]
#                normal = self.normals[i, j, :]
#                ax.plot([x, x+normal[0]], [y, y+normal[1]], [z, z+normal[2]], c='r')
#                ax.scatter(x+normal[0], y+normal[1], z+normal[2], c='r', marker='^')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return fig, ax

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_fp', type=str, required=True, help='path to heightmap file')
    args = parser.parse_args()

    hmap = torch.load(args.map_fp)
    assert hmap.gridZ.shape[0] == hmap.gridZ.shape[1], 'Only works with square heightmaps'
    res = hmap.mapSize[0] / hmap.gridZ.shape[0]
    data = torch.tensor(hmap.gridZ).float()
    w = hmap.mapSize[0] / 2.

    m = HeightMap(data, resolution=res)

    """
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    m.render_3d(fig=fig, ax=ax1)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(-1, 1)
    ax2.imshow(m.get_smoothness(), extent=(-w, w, -w, w))
    ax3.imshow(m.data, extent=(-w, w, -w, w))
    plt.show()
    """

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs[0, 0].imshow(m.data, extent=(-w, w, -w, w))
    axs[0, 1].imshow(m.get_cost(), extent=(-w, w, -w, w), vmin=0., vmax=1.)
    axs[0, 2].imshow(m.get_cost(k_smooth=1.0, k_slope=0.5, k_curvature=0.05), extent=(-w, w, -w, w), vmin=0., vmax=1.)
    axs[1, 0].imshow(m.get_slope(), extent=(-w, w, -w, w))
    axs[1, 1].imshow(m.get_smoothness(), extent=(-w, w, -w, w))
    axs[1, 2].imshow(m.get_curvature(), extent=(-w, w, -w, w))
    axs[2, 0].hist(m.get_slope().numpy().flatten(), bins=20, density=True, histtype='step')
    axs[2, 1].hist(m.get_smoothness().numpy().flatten(), bins=20, density=True, histtype='step')
    axs[2, 2].hist(m.get_curvature().numpy().flatten(), bins=20, density=True, histtype='step')
    axs[0, 0].set_title('Heightmap')
    axs[0, 1].set_title('Cost')
    axs[0, 2].set_title('Weighted Cost')
    axs[1, 0].set_title('Slope')
    axs[1, 1].set_title('Smooth')
    axs[1, 2].set_title('Curvature')
    axs[2, 0].set_title('Slope Hist')
    axs[2, 1].set_title('Smooth Hist')
    axs[2, 2].set_title('Curvature Hist')
    plt.show()

    exit(0)

    idxs = 4. * (torch.rand(2, 10)-0.5)
    vals = m.get_height(idxs)
    normals = m.get_normal(idxs)

    for i, (x, v, n) in enumerate(zip(idxs.T, vals, normals)):
        print("Map value at {} = {}".format(x, v))
        fig, ax = m.render_3d()
        ax.scatter(x[0], x[1], v, c='r')
        ax.plot([x[0], x[0] + n[0]],[x[1], x[1]+n[1]], [v, v+n[2]], c='r')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1, 1)
        plt.show()

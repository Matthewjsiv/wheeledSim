import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt

from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

from wheeledSim.heightmap import HeightMap

class AStarPlanner:
    """
    Simple 8-connected A* planner to get to goal.
    Using a simple L2 distance heuristic for now.
    """
    def __init__(self, heightmap):
        self.heightmap = heightmap
        self.goal = torch.zeros(2)
        self.path = torch.zeros(1, 13)
        self.odom = Odometry()

    def handle_gridmap(self, msg):
        """
        Make heightmap from rosmsg
        """
        hmap_idx = msg.layers.index("height")
        nx = msg.data[hmap_idx].layout.dim[0].size
        ny = msg.data[hmap_idx].layout.dim[1].size
        map_data = torch.tensor(msg.data[hmap_idx].data).view(nx, ny)
        resolution = msg.info.resolution
        self.heightmap = HeightMap(map_data, resolution)

    def handle_goal(self, msg):
        gx = msg.x - self.odom.pose.pose.position.x
        gy = msg.y - self.odom.pose.pose.position.y
        self.goal = torch.tensor([gx, gy])

    def handle_odom(self, msg):
        self.odom = msg

    def path_msg(self):
        """
        Global path for controller to track (needs odom to move to global)
        """
        msg = Path()
        for pt in self.path:
            p_msg = PoseStamped()
            p_msg.header.stamp = rospy.Time.now()
            p_msg.pose.position.x = pt[0] + self.odom.pose.pose.position.x
            p_msg.pose.position.y = pt[1] + self.odom.pose.pose.position.y
            p_msg.pose.position.z = pt[2] + self.odom.pose.pose.position.z
            msg.poses.append(p_msg)
        msg.header.stamp = rospy.Time.now()
        return msg

    def get_grid_position(self, pos):
        """
        Convert a position in m to a grid position (in grid units).
        """
        x = pos[0]
        y = pos[1]
        r = self.heightmap.resolution
        ox = self.heightmap.ox
        oy = self.heightmap.oy
        return torch.tensor([(x/r) + ox, (y/r) + oy]).long()

    def solve(self, start_state, goal_pos, k_smooth=1.0, k_slope=0.5, k_curvature=0.05):
        costmap = self.heightmap.get_cost(k_smooth=k_smooth, k_slope=k_slope, k_curvature=k_curvature)
        start = self.get_grid_position(start_state)
        goal = self.get_grid_position(goal_pos)
        path = self.astar(start, goal, costmap)
        path = torch.stack([x['pos'] for x in path], dim=0)

        """
        plt.imshow(costmap.T, origin='lower')
        plt.plot(path[:, 0], path[:, 1], marker='.', label='path', c='r')
        plt.scatter(start[0], start[1], marker='^', label='start', c='r')
        plt.scatter(goal[0], goal[1], marker='x', label='goal', c='r')
        plt.legend()
        plt.show()
        """

        self.path = self.convert_path(path)

    def convert_path(self, path):
        """
        Convert series of x-y points into a sequence of states.
        Acutally, given the implementation of the PD tracker, all I have to do is expand dims. (get z to plot in pybullet, though)
        """
        r = self.heightmap.resolution
        ox = self.heightmap.ox
        oy = self.heightmap.oy
        pad = torch.zeros(path.shape[0], 8).to(path.device)
        pad[:, 3] = 1. #For valid quaternions.

        x = (path[:, [0]] - ox) * r
        y = (path[:, [1]] - oy) * r
        z = self.heightmap.data[path[:,0], path[:, 1]].unsqueeze(-1)

        return torch.cat([x, y, z, pad], dim=-1)

    def astar(self, start, goal, costmap):
        k_dist = 0.1
        N = costmap.shape[0]
        openlist = [{'pos':start, 'g':0, 'h':k_dist * torch.norm(goal.float() - start.float()), 'f':k_dist * torch.norm(goal.float() - start.float()), 'prev':None}]
        closedlist = []
        node_hash = {}

        #For grids, Nx + y is a collision-free hash.
        def hashcode(x):
            return int(x[0]*N + x[1])

        node_hash[hashcode(start)] = openlist[0]

        nodes = 0

        plt.show(block=False)
        while openlist:
            nodes += 1
            print('Nodes = {}'.format(nodes), end='\r')

            """
            plt.cla()
            plt.imshow(costmap.T, origin='lower')
            print("_____OPEN_____")
            for x in openlist:
                print(x)
                plt.scatter(x['pos'][0], x['pos'][1], marker='.', c='g')
            print("_____CLOSED_____")
            for x in closedlist:
                print(x)
                plt.scatter(x['pos'][0], x['pos'][1], marker='x', c='r')

            plt.scatter(start[0], start[1], marker='^', c='y', label='Start')
            plt.scatter(goal[0], goal[1], marker='*', c='y', label='Goal')
            plt.legend()
            plt.pause(1e-2)
            import pdb;pdb.set_trace()
            """

            u = openlist.pop(0)
            closedlist.append(u)


            if hashcode(u['pos']) == hashcode(goal):
                break

            neighbors = self.expand(u['pos'], N)
            for v in neighbors:
                g = u['g'] + costmap[v[0], v[1]] + k_dist * torch.norm(v.float() - u['pos'].float())
                h = k_dist * torch.norm(goal.float() - v.float())
                f = g + h
                prev = u
                if (hashcode(v) not in node_hash.keys()):
                    node = {'pos':v, 'g':g, 'h':h, 'f':f, 'prev':u['pos']}
                    openlist.append(node)
                    node_hash[hashcode(v)] = node
                elif node_hash[hashcode(v)]['f'] > f:
                    node_hash[hashcode(v)]['g'] = g
                    node_hash[hashcode(v)]['h'] = h
                    node_hash[hashcode(v)]['f'] = f
                    node_hash[hashcode(v)]['prev'] = u['pos']

            openlist = sorted(openlist, key=lambda x:x['f'])

        #Get path
        print('getting path...')
        path = [closedlist[-1]]
        while path[-1]['prev'] is not None:
            path.append(node_hash[hashcode(path[-1]['prev'])])

        return reversed(path)

    def expand(self, x, N):
        diffs = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
#        diffs = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1], [-2, -2], [-2, 0], [-2, 2], [0, -2], [0, 2], [2, -2], [2, 0], [2, 2]])
        neighbors = x.unsqueeze(0) + diffs
        mask = torch.any((neighbors < 0) | (neighbors >= N), dim=-1)
        return neighbors[~mask]

    def viz(self):
        plt.cla()
        plt.imshow(planner.heightmap.data, origin='lower', extent=(-self.heightmap.width, self.heightmap.width, -self.heightmap.height, self.heightmap.height))
        plt.scatter(self.goal[0], self.goal[1], marker='^', label='goal')
        plt.plot(self.path[:, 0], self.path[:, 1], marker='.', label='path')
        plt.legend()
        plt.pause(1e-2)

if __name__ == '__main__':
    hmap_init = HeightMap(torch.zeros(100, 100), 0.1)
    planner = AStarPlanner(hmap_init)

    rospy.init_node("local_planner")

    hmap_sub = rospy.Subscriber("/local_map", GridMap, planner.handle_gridmap)
    odom_sub = rospy.Subscriber("/odom", Odometry, planner.handle_odom)
    path_pub = rospy.Publisher("/planner/path", Path)

    delay = 1.

    rate = rospy.Rate(2)

    print('waiting {}s for goal...'.format(delay))
    for i in range(int(delay * 2)):
        rate.sleep()

    planner.solve(torch.zeros(13), torch.tensor([2., 1.]))
    planner.viz()
    plt.show()

    print('Publishing path...')
#    for _ in range(10):
    path_pub.publish(planner.path_msg())
#        rate.sleep()

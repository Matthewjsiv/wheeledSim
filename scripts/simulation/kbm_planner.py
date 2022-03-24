import rospy
import torch
import matplotlib.pyplot as plt
import copy

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import Quaternion

from wheeledSim.rosSimController import rosSimController
from wheeledSim.models.kbm_dynamics import KBMDynamics
from wheeledSim.models.kinematic_bicycle_model import KBMKinematics
from wheeledSim.models.transfer_functions import ARXTransferFunction
from wheeledSim.heightmap import HeightMap
from wheeledSim.util import quat_to_yaw

class KBMBasePlanner:
    """
    Plannning class that uses a KBM with primitives to expand nodes.
    Note that since we're using a model, nodes need to store actions/states.
    """
    def __init__(self, model, primitives, hmap_params, dt=0.1, odom_topic='/odom', heightmap_topic='/heightmap', goal_topic='/goal', relative_goal=False):
        """
        Args:
            model: The car model through which to plan
            primitives: A list of torch tensors that pre-define action sequences. Each primitive should be batched as [timedim x actdim]
            hmap_params: The expected params of the heightmap. Expects a dict containing senseResolution and senseDim
            odom_topic: The rostopic to listen for state on
            heightmap_topic: The rostopic to listen for the heightmap on
            goal_topic: The topic to listen for (a GLOBAL) goal position on
        """
        self.model = model
        self.primitives = primitives
        self.heightmap_params = hmap_params
        self.resolution = hmap_params["senseDim"][0] / hmap_params["senseResolution"][0]
        self.sensor_translation = torch.tensor(hmap_params["sensorPose"][0][:2]).float() #Note that we dont allow sensor rotation
        self.dt = dt

        self.odom_topic = odom_topic
        self.heightmap_topic = heightmap_topic
        self.goal_topic = goal_topic

        self.position = torch.zeros(3)
        self.goal = torch.zeros(2)
        self.relative_goal = self.goal - self.position[:2]
        self.heightmap = HeightMap(torch.zeros(*hmap_params["senseResolution"]), self.resolution)
        self.use_relative_goal = relative_goal

        self.model_predictions = None
        self.costs = torch.zeros(len(self.primitives))
        self.best_seq = torch.zeros(1, 2)

    def handle_odom(self, msg):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        self.position[2] = quat_to_yaw(msg.pose.pose.orientation)
        if not self.use_relative_goal:
            self.get_relative_goal()

    def handle_heightmap(self, msg):
        height_idx = msg.layers.index('height')
        data = torch.tensor(msg.data[height_idx].data).float()
        rowwidth = msg.data[height_idx].layout.dim[1].size
        hmap = data.view(-1, rowwidth).T #TODO: This is flipped between ATV and simulation. Take out the transpose for pybullet.
        self.heightmap = HeightMap(hmap, self.resolution, k_smooth=0.5, k_slope=0.05, k_curvature=0.5)

    def handle_goal(self, msg):
        self.goal[0] = msg.pose.position.x
        self.goal[1] = msg.pose.position.y
        if self.use_relative_goal:
            self.relative_goal = self.goal.clone()
        else:
            self.get_relative_goal()

    def get_relative_goal(self):
        """
        Get the goal relative to the current position.
        I.e. translate then rotate
        """
        relative_position = self.goal - self.position[:2]
        yaw = -self.position[2]
        R = torch.tensor([[torch.cos(yaw), -torch.sin(yaw)],[torch.sin(yaw), torch.cos(yaw)]])
        self.relative_goal = torch.matmul(R, relative_position).squeeze()

    def plan(self):
        """
        Use some kind of planning to generate a sequence of actions (of arbitrary length) and store in self.best_seq
        """
        pass

    def render(self, fig=None, axs=None):
        pass


class KBMMPCPlanner(KBMBasePlanner):
    def plan(self):
        """
        Dead simple. Try all the primitives and take the one with the minimum cost+cost_to_go
        """
        res = []
        for seq in self.primitives:
            temp_model = copy.deepcopy(self.model) #This is really bad. Fix later.
            traj = []
#            cstate = self.position.clone()
            cstate = torch.zeros(3) #Everything's in local frame, so starting at 0,0,0rad is fine
            for act in seq:
                cstate = temp_model.forward(cstate, act)
                traj.append(cstate.clone())
            traj = torch.stack(traj, dim=0)
            res.append(traj)
        res = torch.stack(res, dim=0)

        #Account for the transform of the sensor map.
        hmap_pos = res[:, :, :2] - self.sensor_translation.unsqueeze(0).unsqueeze(0)

        obstacle_cost = self.heightmap.get_cost([hmap_pos[:, :, 0], hmap_pos[:, :, 1]]).sum(dim=-1)
        cost_to_go = torch.linalg.norm(res[:, -1, :2] - self.relative_goal, dim=-1)
        costs = obstacle_cost + 5. * cost_to_go

        self.model_predictions = res
        self.costs = costs
        self.best_seq = self.primitives[self.costs.argmin()]

    def render(self, fig=None, axs=None):
        if fig is None or axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(9, 4))

        xmin = -self.heightmap_params["senseDim"][0]/2 + self.sensor_translation[0]
        ymin = -self.heightmap_params["senseDim"][1]/2 + self.sensor_translation[1]
        xmax = xmin + 2*self.heightmap_params["senseDim"][0]/2
        ymax = ymin + 2*self.heightmap_params["senseDim"][1]/2
        axs[0].imshow(self.heightmap.data, origin='lower', extent=(xmin, xmax, ymin, ymax), vmin=-2.0, vmax=2.0)
        axs[1].imshow(self.heightmap.cost, origin='lower', extent=(xmin, xmax, ymin, ymax), vmin=0.0, vmax=1.0)

        for ax in axs:
            ax.scatter(0., 0., c='g', marker='>', label='Current')
            ax.scatter(self.relative_goal[0], self.relative_goal[1], c='r', marker='x', label='Goal')

            if self.model_predictions is not None:
                for cost, pred in zip(self.costs, self.model_predictions):
                    ax.plot(pred[:, 0], pred[:, 1], marker='.')
                    ax.text(pred[-1, 0], pred[-1, 1], '{:.2f}'.format(cost), color='w')

        axs[0].legend()
        axs[1].legend()
        axs[0].set_title('Heightmap')
        axs[1].set_title('Costmap')

        return fig, axs

class KBMAstarPlanner(KBMBasePlanner):
    def __init__(self, model, primitives, hmap_params, dt=0.1, max_itrs=10, odom_topic='/odom', heightmap_topic='/heightmap', goal_topic='/goal', relative_goal=False):
        super(KBMAstarPlanner, self).__init__(model, primitives, hmap_params, dt, odom_topic, heightmap_topic, goal_topic, relative_goal)
        self.max_itrs = max_itrs
        self.root = None

    def plan(self):
        """
        Implement some kind of A* with motion primitives (its really just a tree)
        TODO: Again, it's super garbage to copy the model state for every step in the planner
        """
#        if torch.linalg.norm(self.relative_goal) > 0.1:
#            import pdb;pdb.set_trace()

        state = torch.zeros(3)
        self.root = AstarNode(g=torch.tensor(0.), h=self.cost_to_go(state), state=state, model=copy.deepcopy(self.model), prev=None, prev_acts=torch.zeros(1, 2))

        openlist = [self.root]
        closedlist = []

        for i in range(self.max_itrs):
            curr = openlist.pop(0)
            children = self.expand(curr)
            curr.children = children
            openlist.extend(children)
            closedlist.append(curr)
            openlist.sort(key=lambda x:x.f)

        best = min(closedlist, key=lambda x:x.f)
        curr = best
        seq = torch.zeros(0, 2)
        while curr.prev is not None:
            seq = torch.cat([curr.prev_acts, seq], dim=0)
            curr = curr.prev

        #Store these and the tree for plotting
        self.best_node = best
        self.best_seq = torch.zeros(1, 2) if len(seq) == 0 else seq

    def expand(self, root):
        """
        Run all the primitives from this node
        """
        nodes = []
        for seq in self.primitives:
            temp_model = copy.deepcopy(root.model) #This is really bad. Fix later.
            traj = []
            cstate = root.state
            for act in seq:
                cstate = temp_model.forward(cstate, act)
                traj.append(cstate.clone())
            traj = torch.stack(traj, dim=0)

            hmap_pos = traj[:, :2] - self.sensor_translation.unsqueeze(0)
            obstacle_cost = self.heightmap.get_cost([hmap_pos[:, 0], hmap_pos[:, 1]]).sum(dim=-1)
            ctg = self.cost_to_go(traj[-1])

            new_node = AstarNode(g=root.g + obstacle_cost, h=ctg, state=traj[-1], model=temp_model, prev=root, prev_acts=seq)

            if torch.linalg.norm(traj[-1, :2] - root.state[:2]) > 0.25:
                nodes.append(new_node)

        return nodes

    def cost_to_go(self, state):
        return 3. * torch.linalg.norm(state[:2] - self.relative_goal)

    def render(self, fig=None, axs=None):
        if fig is None or axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(9, 4))

        xmin = -self.heightmap_params["senseDim"][0]/2 + self.sensor_translation[0]
        ymin = -self.heightmap_params["senseDim"][1]/2 + self.sensor_translation[1]
        xmax = xmin + 2*self.heightmap_params["senseDim"][0]/2
        ymax = ymin + 2*self.heightmap_params["senseDim"][1]/2
        axs[0].imshow(self.heightmap.data, origin='lower', extent=(xmin, xmax, ymin, ymax), vmin=-2.0, vmax=2.0)
        axs[1].imshow(self.heightmap.cost, origin='lower', extent=(xmin, xmax, ymin, ymax), vmin=0.0, vmax=1.0)

        for ax in axs:
            ax.scatter(0., 0., c='g', marker='>', label='Current')
            ax.scatter(self.relative_goal[0], self.relative_goal[1], c='r', marker='x', label='Goal')

            if self.root is not None:
                #Implement a tree search to get endpts
                self.plot_tree(self.root, fig, axs)
                #Plot the selected path
                states = []
                curr = self.best_node
                while curr is not None:
                    states.append(curr.state)
                    curr = curr.prev

                for ax in axs:
                    xs = [x[0] for x in states]
                    ys = [x[1] for x in states]
                    ax.plot(xs, ys, c='r', marker='.')
                    ax.text(xs[0], ys[0], '{:.2f}'.format(self.best_node.f), color='w')

        axs[0].legend()
        axs[1].legend()
        axs[0].set_title('Heightmap')
        axs[1].set_title('Costmap')

        return fig, axs

    def plot_tree(self, root, fig, axs):
        """
        (recursively) plot the search tree
        """
        for child in root.children:
            prev_state = root.state
            curr_state = child.state
            for ax in axs:
                ax.plot([prev_state[0], curr_state[0]], [prev_state[1], curr_state[1]], c='b')

            self.plot_tree(child, fig, axs)

        return fig, axs

class AstarNode:
    def __init__(self, g, h, state, model, prev, prev_acts, children=[]):
        """
        Info here should contain the state
        Prev should contain a pointer to the predecessor as well as the sequence
        """
        self.g = g
        self.h = h
        self.state = state
        self.model = model
        self.prev = prev
        self.prev_acts = prev_acts
        self.children = children

    @property
    def f(self):
        return self.g + self.h

    def __repr__(self):
        return "NODE:\tstate: {}\n\tf: {:.2f}\n\tg: {:.2f}\n\th: {:.2f}\n\tacts: {}\n".format(self.state, self.f.item(), self.g.item(), self.h.item(), self.prev_acts[0] if len(self.prev_acts) > 0 else self.prev_acts)

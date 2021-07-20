import rospy
import torch
import matplotlib.pyplot as plt
import copy

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap

from wheeledSim.rosSimController import rosSimController
from wheeledSim.models.kbm_dynamics import KBMDynamics
from wheeledSim.models.kinematic_bicycle_model import KBMKinematics
from wheeledSim.models.transfer_functions import ARXTransferFunction
from wheeledSim.heightmap import HeightMap
from wheeledSim.util import quat_to_yaw

class KBMMPCPlanner:
    """
    Plannning class that uses a KBM with primitives to expand nodes.
    Note that since we're using a model, nodes need to store actions/states.
    """
    def __init__(self, model, primitives, hmap_params, dt=0.1, odom_topic='/odom', heightmap_topic='/heightmap', goal_topic='/goal'):
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
        self.dt = dt

        self.odom_topic = odom_topic
        self.heightmap_topic = heightmap_topic
        self.goal_topic = goal_topic

        self.position = torch.zeros(3)
        self.heightmap = HeightMap(torch.zeros(*hmap_params["senseResolution"]), self.resolution)
        self.goal = torch.zeros(2)
        self.relative_goal = self.goal - self.position[:2]
        self.model_predictions = None
        self.costs = torch.zeros(len(self.primitives))
        self.best_seq = torch.zeros(1, 2)

    def handle_odom(self, msg):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        self.position[2] = quat_to_yaw(msg.pose.pose.orientation)
        self.get_relative_goal()

    def handle_heightmap(self, msg):
        height_idx = msg.layers.index('height')
        data = torch.tensor(msg.data[height_idx].data).float()
        rowwidth = msg.data[height_idx].layout.dim[1].size
        hmap = data.view(-1, rowwidth)
        self.heightmap = HeightMap(hmap, self.resolution)

    def handle_goal(self, msg):
        self.goal[0] = msg.pose.position.x
        self.goal[1] = msg.pose.position.y
        self.get_relative_goal()

    def get_relative_goal(self):
        """
        Get the goal relative to the current position.
        I.e. translate then rotate
        """
        yaw = -self.position[2]
        R = torch.tensor([[torch.cos(yaw), -torch.sin(yaw)],[torch.sin(yaw), torch.cos(yaw)]])
        relative_position = self.goal - self.position[:2]
        self.relative_goal = torch.matmul(R, relative_position).squeeze()

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

        obstacle_cost = self.heightmap.get_cost([res[:, :, 0], res[:, :, 1]]).sum(dim=-1)
        cost_to_go = torch.linalg.norm(res[:, -1, :2] - self.relative_goal, dim=-1)
        costs = obstacle_cost + cost_to_go

        self.model_predictions = res
        self.costs = costs
        self.best_seq = self.primitives[self.costs.argmin()]

    def render(self, fig=None, axs=None):
        if fig is None or axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(9, 4))

        xmin = -self.heightmap_params["senseDim"][0]
        ymin = -self.heightmap_params["senseDim"][1]
        xmax = -xmin
        ymax = -ymin
        axs[0].imshow(self.heightmap.data, origin='lower', extent=(xmin, xmax, ymin, ymax))
        axs[1].imshow(self.heightmap.cost, origin='lower', extent=(xmin, xmax, ymin, ymax))

        for ax in axs:
            ax.scatter(0., 0., c='g', marker='>', label='Current')
            ax.scatter(self.relative_goal[0], self.relative_goal[1], c='r', marker='x', label='Goal')

            if self.model_predictions is not None:
                for cost, pred in zip(self.costs, self.model_predictions):
                    ax.plot(pred[:, 0], pred[:, 1])
                    ax.text(pred[-1, 0], pred[-1, 1], '{:.2f}'.format(cost), color='w')

        axs[0].legend()
        axs[1].legend()

        return fig, axs

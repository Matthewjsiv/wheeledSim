import torch

from wheeledSim.models.kinematic_bicycle_model import KBMKinematics
from wheeledSim.models.transfer_functions import ARXTransferFunction

class KBMDynamics:
    """
    Combine transfer func with a kinematic bicycle model in order to get dynamics.
    """
    def __init__(self, model, throttle_tf, steer_tf, device='cpu'):
        self.model = model
        self.steer_tf = steer_tf
        self.throttle_tf = throttle_tf
        
        self.state_dim = self.model.state_dim
        self.control_dim = self.model.control_dim

        self.prev_vel = torch.tensor(0.)
        self.prev_yaw = torch.tensor(0.)

        self.device = device

    def forward(self, state, action, dt=0.1):
        """
        Get previous velocity/realtive yaw from model
        """
        throttle, steer = action.swapdims(-1, 0)

        vel = self.throttle_tf.forward(self.prev_vel, throttle)
        yaw = self.steer_tf.forward(self.prev_yaw, steer)

        res = self.model.forward_dynamics(state, torch.stack([vel, yaw], dim=-1), dt=dt)

        self.prev_vel = vel.clone()
        self.prev_yaw = yaw.clone()

        return res

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.throttle_tf = self.throttle_tf.to(device)
        self.steer_tf = self.steer_tf.to(device)
        self.prev_vel = self.prev_vel.to(device)
        self.prev_yaw = self.prev_yaw.to(device)
        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    kbm = KBMKinematics()
#    throttle_tf = ARXTransferFunction()
#    steer_tf = ARXTransferFunction()
    throttle_tf = torch.load('f1p0_throttle.pt')
    steer_tf = torch.load('f1p0_steer.pt')

    dynamic_kbm = KBMDynamics(kbm, throttle_tf, steer_tf)

    seqs = torch.ones(30, 2)
    seqs[15:, 1] *= -0.5
    state = torch.zeros(dynamic_kbm.state_dim)
    res = [state]
    states = [torch.zeros(2)]
    for t in range(seqs.shape[0]):
        cs = res[-1]
        res.append(dynamic_kbm.forward(cs, seqs[t]))
        states.append(torch.stack([dynamic_kbm.prev_vel, dynamic_kbm.prev_yaw]))

    traj = torch.stack(res, dim=0)
    states = torch.stack(states, dim=0)

    #Plot traj, state vars and cmds
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))

    axs[0].set_title('Trajectory')
    axs[1].set_title('State Vars')
    axs[2].set_title('Cmds')

    axs[0].plot(traj[:, 0], traj[:, 1], label='Traj')
    axs[1].plot(states[:, 0], label='Velocity')
    axs[1].plot(states[:, 1], label='Relative Yaw')
    axs[2].plot(seqs[:, 0], label='Throttle')
    axs[2].plot(seqs[:, 1], label='Steer')

    axs[0].legend()
    axs[0].set_aspect('equal')
    axs[1].legend()
    axs[2].legend()
    axs[2].set_ylim(-1.1, 1.1)

    plt.show()


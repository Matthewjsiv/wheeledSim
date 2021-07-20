import torch
import matplotlib.pyplot as plt
import argparse

from torch import optim

import numpy as np

class ARXTransferFunction:
    """
    Transfer function that is linear in the last k states (targets) and actions.
    """
    def __init__(self, buf_size=3, only_last_state=False, order=1, device='cpu'):
        """
        Args:
            use_last_state: If True, just use lase item in buf (TODO: Change the forward func.)
            order: Square the input up to this many times.
        """
        #Most recent is the front
        self.xbuf = torch.zeros(buf_size)
        self.ubuf = torch.zeros(buf_size)

        self.only_last_state = only_last_state
        self.order = order

        self.Kx = torch.zeros(order, 1) if only_last_state else torch.zeros(order, buf_size)
        self.Ku = torch.zeros(order, 1) if only_last_state else torch.zeros(order, buf_size)

        self.device = device

    def set_grad(self, grad):
        self.Kx.requires_grad = grad
        self.Ku.requires_grad = grad

    def reset(self):
        """
        Clear bufs
        """
        self.xbuf = torch.zeros(buf_size)
        self.ubuf = torch.zeros(buf_size)

    def batch_forward(self, states, cmds):
        """
        Stateless version of forward for param fitting.
        Expects states/cmds batched as [batch x time]
        """
        if self.only_last_state:
            return sum([self.Kx[i, -1] * states[:, -1].pow(i+1) + self.Ku[i, -1] * cmds[:, -1] for i in range(self.order)])
        else:
            return sum([(self.Kx[i, :] * states.pow(i+1)).sum(dim=-1) + (self.Ku[i, :] * cmds.pow(i+1)).sum(dim=-1) for i in range(self.order)])

    def forward(self, state, cmd):
        """
        Here, expects both state and cmd to be a single value
        """
        self.xbuf = torch.cat([state.unsqueeze(0), self.xbuf[:-1]], dim=0)
        self.ubuf = torch.cat([cmd.unsqueeze(0), self.ubuf[:-1]], dim=0)

        if self.only_last_state:
            return sum([self.Kx[i, -1] * self.xbuf[-1].pow(i+1) + self.Ku[i, -1] * self.ubuf[-1] for i in range(self.order)])
        else:
            return sum([(self.Kx[i, :] * self.xbuf.pow(i+1)).sum(dim=-1) + (self.Ku[i, :] * self.ubuf.pow(i+1)).sum(dim=-1) for i in range(self.order)])

    def __repr__(self):
        return "Kx: {}\nKu: {}\nxbuf: {}\nubuf: {}".format(self.Kx, self.Ku, self.xbuf, self.ubuf)

    def to(self, device):
        self.device = device
        self.Kx = self.Kx.to(device)
        self.Ku = self.Ku.to(device)
        self.xbuf = self.xbuf.to(device)
        self.ubuf = self.ubuf.to(device)
        return self

def generate_batch_data(buf, N=3):
    """
    Recall data is batched as [traj x time x feat]
    Output as X: [batch x time x feat]
              Y: [batch x feat]
    """
    #Loop around outer dim for different trajlens

    states = buf['sysid_labels']
    controls = buf['action']

    states_out = []
    states_in = []
    controls_in = []

    for x, u in zip(states, controls):
        x_out = x[N:]
        x_in = torch.stack([x[i:-(N-i)] for i in reversed(range(N))], dim=-2)
        u_in = torch.stack([u[i:-(N-i)] for i in reversed(range(N))], dim=-2)

        states_out.append(x_out)
        states_in.append(x_in)
        controls_in.append(u_in)

    return {'X':{'state':torch.cat(states_in, dim=0), 'control':torch.cat(controls_in, dim=0)}, 'Y':torch.cat(states_out, dim=0)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fp', type=str, required=False, help="Path to data to eval")
    parser.add_argument('--throttle_tf', type=str, required=False, default=None, help="Path to data to eval")
    parser.add_argument('--steer_tf', type=str, required=False, default=None, help="Path to data to eval")
    args = parser.parse_args()

    buf_size = 2

    prefix = 'systematic_data'
    frict_vals = np.linspace(.5,2.5,10)
    # slope_vals = np.linspace(-.5,.5,10)
    slope_vals = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]
    for frict in frict_vals:
        for slope in slope_vals:

            fpath = prefix + '/s' + str(slope) + 'f' + str(frict)
            trajdata = torch.load(fpath)

            data = generate_batch_data(trajdata, buf_size)

            #Train
            if args.throttle_tf is None or args.steer_tf is None:
                throttle_tf = ARXTransferFunction(buf_size=buf_size, only_last_state=False, order=3)
                throttle_tf.set_grad(True)
                throttle_opt = optim.LBFGS([throttle_tf.Kx, throttle_tf.Ku])

                steer_tf = ARXTransferFunction(buf_size=buf_size, only_last_state=False, order=3)
                steer_tf.set_grad(True)
                steer_opt = optim.LBFGS([steer_tf.Kx, steer_tf.Ku])

                def throttle_closure():
                    if torch.is_grad_enabled():
                        throttle_opt.zero_grad()
                    preds = throttle_tf.batch_forward(data['X']['state'][:, :, 0], data['X']['control'][:, :, 0])
                    err = data['Y'][:, 0] - preds
                    loss = err.pow(2).mean()
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                def steer_closure():
                    if torch.is_grad_enabled():
                        steer_opt.zero_grad()
                    preds = steer_tf.batch_forward(data['X']['state'][:, :, 1], data['X']['control'][:, :, 1])
                    err = data['Y'][:, 1] - preds
                    loss = err.pow(2).mean()
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                epochs = 100

                for e in range(epochs):
                    throttle_opt.step(throttle_closure)
                    steer_opt.step(steer_closure)

                    #Get MSE to print
                    with torch.no_grad():
                        throttle_mse = throttle_closure()
                        steer_mse = steer_closure()

                    print('EPOCH {}: Throttle MSE = {:.6f}, Steer MSE = {:.6f}'.format(e+1, throttle_mse.item(), steer_mse.item()))

                torch.set_printoptions(sci_mode=False, linewidth=1000)
                throttle_tf.set_grad(False)
                steer_tf.set_grad(False)
                print("THROTTLE TF:\n", throttle_tf)
                print("STEER TF:\n", steer_tf)

                torch.save(throttle_tf, prefix + '/tfs' + '/s' + str(slope) + 'f' + str(frict)  + '_throttle.pt')
                torch.save(steer_tf, prefix + '/tfs' + '/s' + str(slope) + 'f' + str(frict) + '_steer.pt')
            else:
                throttle_tf = torch.load(args.throttle_tf)
                steer_tf = torch.load(args.steer_tf)

            # #Viz
            # for _ in range(5):
            #     tidx = torch.randint(len(trajdata['action']), (1,)).squeeze()
            #     vel = trajdata['sysid_labels'][tidx][:, 0]
            #     delta = trajdata['sysid_labels'][tidx][:, 1]
            #     throttle = trajdata['action'][tidx][:, 0]
            #     steer = trajdata['action'][tidx][:, 1]
            # 
            #     vpred = [vel[0]]
            #     dpred = [delta[0]]
            #     throttle_tf.reset()
            #     steer_tf.reset()
            #
            #     for v, d, t, s in zip(vel, delta, throttle, steer):
            #         vpred.append(throttle_tf.forward(vpred[-1], t))
            #         dpred.append(steer_tf.forward(dpred[-1], s))
            #
            #     vpred = torch.tensor(vpred[1:])
            #     dpred = torch.tensor(dpred[1:])
            #
            #     throttle_error = (vpred - vel).pow(2).mean().sqrt()
            #     steer_error = (dpred - steer).pow(2).mean().sqrt()
            #
            #     print("_" * 30)
            #     print("Vel error = {:.4f}".format(throttle_error))
            #     print("Steer error = {:.4f}".format(steer_error))
            #
            #     plt.plot(vel, label='vgt')
            #     plt.plot(delta, label='dgt')
            #     plt.plot(vpred, label='vpred')
            #     plt.plot(dpred, label='dpred')
            #     plt.legend()
            #     plt.show()

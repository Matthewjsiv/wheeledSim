import torch
import matplotlib.pyplot as plt
import argparse

from torch import optim

class ARXTransferFunction:
    """
    Transfer function that is linear in the last k states (targets) and actions.
    """
    def __init__(self, buf_size=3, device='cpu'):
        self.Kx = torch.zeros(buf_size)
        self.Ku = torch.zeros(buf_size)

        #Most recent is the front
        self.xbuf = torch.zeros(buf_size)
        self.ubuf = torch.zeros(buf_size)
        self.device = device

    def set_grad(self, grad):
        self.Kx.requires_grad = grad
        self.Ku.requires_grad = grad

    def batch_forward(self, states, cmds):
        """
        Stateless version of forward for param fitting.
        Expects states/cmds batched as [batch x time]
        """
        return (self.Kx * states).sum(dim=-1) + (self.Ku * cmds).sum(dim=-1)

    def forward(self, state, cmd):
        """
        Here, expects both state and cmd to be a single value
        """
        self.xbuf = torch.cat([state.unsqueeze(0), self.xbuf[:-1]], dim=0)
        self.ubuf = torch.cat([cmd.unsqueeze(0), self.ubuf[:-1]], dim=0)

        return (self.Kx * self.xbuf).sum(dim=-1) + (self.Ku * self.ubuf).sum(dim=-1)

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

    states = buf['sysid_labels']
    controls = buf['action']

    states_out = states[:, N:]
    states_in = torch.stack([states[:, i:-(N-i)] for i in reversed(range(N))], dim=2)
    controls_in = torch.stack([controls[:, i:-(N-i)] for i in reversed(range(N))], dim=2)

    return {'X':{'state':states_in.flatten(end_dim=-3), 'control':controls_in.flatten(end_dim=-3)}, 'Y':states_out.flatten(end_dim=-2)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fp', type=str, required=True, help="Path to data to eval")
    args = parser.parse_args()

    buf_size = 5
    data = torch.load(args.data_fp)

    #TODO: Convert data to batch format
    data = generate_batch_data(data, buf_size)

    throttle_tf = ARXTransferFunction(buf_size=buf_size)
    throttle_tf.set_grad(True)
    throttle_opt = optim.LBFGS([throttle_tf.Kx, throttle_tf.Ku])

    steer_tf = ARXTransferFunction(buf_size=buf_size)
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

#    for x in torch.linspace(0, 1, 6):
#        throttle_tf.forward(x, -x)
#        print(throttle_tf)

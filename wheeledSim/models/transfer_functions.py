import torch

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

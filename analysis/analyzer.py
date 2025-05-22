"""class for fixed point analysis"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable


class FixedPoint(object):
    def __init__(self, model, device, gamma=0.01, speed_tor=1e-03, max_epochs=100000,
                 lr_decay_epoch=20000):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.speed_tor = speed_tor
        self.max_epochs = max_epochs
        self.lr_decay_epoch = lr_decay_epoch

        self.model.eval()

    def calc_speed(self, h, x):
        # h: (1,hid), x: (1,in)
        x = x.squeeze(0)
        h = h.squeeze(0)

        U = self.model.U
        W = self.model.W
        b = self.model.b.squeeze(1)

        pre = U @ x + W @ h + b
        if self.model.nonlinearity == 'relu':
            activated = F.relu(pre)
        else:
            activated = pre 
        return torch.norm(activated - h)
    


    def find_fixed_point(self, init_hidden, x_tensor, view=False):
        new_h = init_hidden.clone().detach().to(self.device)
        gamma = self.gamma
        for i in range(self.max_epochs+1):
            h = new_h.clone().detach().requires_grad_(True)
            speed = self.calc_speed(h, x_tensor)
            # print(f'epoch: {i}, speed={speed.item()}')
            if speed.item() < self.speed_tor:
                break
            grad_h = torch.autograd.grad(speed, h)[0]
            new_h = (h - gamma * grad_h).detach()
            if i>0 and i % self.lr_decay_epoch == 0:
                gamma *= 0.5
        if i == self.max_epochs:
            print(f"Forced exit at max_epochs, final speed={speed.item():.6f}")
        return new_h, speed.item(), speed.item() < self.speed_tor


    def calc_jacobian(self, fixed_point, const_signal_tensor):
        # fixed_point: shape (1, n_hidden) → we want (n_hidden, 1)
        fixed_point = fixed_point.squeeze(0).unsqueeze(1).to(self.device)  # shape: [n_hidden, 1]
        fixed_point.requires_grad = True

        x = const_signal_tensor[0, 0, :].view(-1, 1).to(self.device)  # shape: [input_dim, 1]

        U = self.model.U.to(self.device)
        W = self.model.W.to(self.device)
        b = self.model.b.to(self.device)

        # Compute pre-activations
        pre_activates = U @ x + W @ fixed_point + b  # shape: [n_hidden, 1]

        # Apply nonlinearity
        if self.model.nonlinearity == 'relu':
            activated = F.relu(pre_activates)
        else:
            activated = pre_activates

        # Compute Jacobian
        jacobian = torch.zeros(self.model.n_hidden, self.model.n_hidden).to(self.device)
        for i in range(self.model.n_hidden):
            grad_output = torch.zeros_like(activated)
            grad_output[i] = 1.0
            grads = torch.autograd.grad(activated, fixed_point, grad_outputs=grad_output, retain_graph=True)[0]
            jacobian[i, :] = grads.view(-1)

        return jacobian.cpu().numpy()

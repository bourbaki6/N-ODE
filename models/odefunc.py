#---N-ODE treats transformative of hidden state as a 
#   continuous dynamicla system than seq of discrete layer transformations---#

#--- Resnet is a N-ODE with a fixed euler solver and dt = 1---#

#---Arch.: I use Tanh bacause it's smooth everywhere
# ReLU not smooth at 0 and causes stiffness; and 
# bounds outputs to (-1, 1) preventing explozive gradients---#

#---1 ODEFunction  replaces n Resnet residual blocks
#   param cnt. is fixed, dependent on no of solevr steps---#

import torch 
import torch.nn as nn



class ODEFunction(nn.Module):

    def __init__(self, hidden_dim: int =64):
        
        super().__init__()

        self.hidden_dim = hidden_dim

        #--- +1: time input conc. to h---#
        self.net = nn.Sequential(
            nn.Linear(hidden_dim +1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()
        self.nfe = 0

    def _init_weights(self):

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean = 0.0, std= 0.01)
                nn.init.zeros_(layer.bias)

    
    def forward(self, t: torch.Tensor, h:torch.Tensor) -> torch.Tensor:

        #--- calc. dh/dt for a given (t, h)---#

        self.nfe += 1

        if t.dim() == 0:
            t = t.unsqueeze(0)

        t_expanded = t.expand(h.shape[0], 1)

        h_t = torch.cat([h, t_expanded], dim = 1)

        return self.net(h_t)
    
    def reset_nfe(self):
        #--- resetting counter between forward passes---#
        self.nfe = 0


if __name__ == "__main__":
    func = ODEFunction(hidden_dim=64)

    batch_size = 8
    h = torch.randn(batch_size, 64)
    t = torch.tensor(0.0)

    out = func(t, h)

    print("\n Output shape:", out.shape)
    print("\n NFE:", func.nfe)
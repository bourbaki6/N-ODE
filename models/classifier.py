#---Arch.: 
#       input: linear -> GroupNorm -> Tanh
#       ode_block: ODEFunction integrated by the chosen solver
#       output: GroupNorm -> Linear ----# 
import torch
import torch.nn as nn
from models.odefunc import ODEFunction
from .ode_block import ODEBlock

class NeuralODEClassifier(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 10,
        solver: str = "rk4",
        num_steps: int = 10,
        input_dim: int = 784,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.solver = solver

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GroupNorm(8, hidden_dim),   
            nn.Tanh(),                      
        )

        self.odefunc = ODEFunction(hidden_dim=hidden_dim)
        self.ode_block = ODEBlock(
            odefunc = self.odefunc,
            solver = solver,
            num_steps = num_steps,
        )

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.view(x.size(0), -1)

        h0 = self.input_proj(x)
        h1 = self.ode_block(h0)

        logits = self.output_proj(h1)

        return torch.log_softmax(logits, dim = -1)
    

    def set_solver(self, solver: str, num_steps: int = None):
        self.solver = solver
        self.ode_block.set_solver(solver, num_steps)

    @property
    def nfe(self) -> int:
        return self.ode_block.nfe
    
    def count_parameters(self) -> dict:

        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
 
        return {
            "input_proj": count(self.input_proj),
            "odefunc": count(self.odefunc),       
            "output_proj": count(self.output_proj),
            "total": count(self),
        }

import torch
import torch.nn as nn
 
 
class ResidualBlock(nn.Module):
 
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),                           
            nn.Linear(hidden_dim, hidden_dim),
        )
 
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.net(h)   
 
class ResNetBaseline(nn.Module):
 
    def __init__(
        self,
        hidden_dim: int = 64,
        num_blocks: int = 6,
        num_classes: int = 10,
        input_dim: int = 784,
    ):
        super().__init__()
 
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
 
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.Tanh(),
        )
 
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
 
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        x = x.view(x.size(0), -1)      
        h = self.input_proj(x)         
        h = self.blocks(h)              
        logits = self.output_proj(h)     
        return torch.log_softmax(logits, dim=-1)
 
    def count_parameters(self) -> dict:

        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
 
        block_params = sum(count(b) for b in self.blocks)
        return {
            "input_proj": count(self.input_proj),
            "blocks_total": block_params,
            "per_block":  block_params // self.num_blocks,
            "output_proj":  count(self.output_proj),
            "total": count(self),
        }
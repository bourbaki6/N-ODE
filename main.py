
import torch
from models.odefunc import ODEFunction
from models.ode_block import ODEBlock


def main():
    
    func = ODEFunction(hidden_dim = 64)
    ode_block = ODEBlock(func, solver = "rk4", num_steps = 10)

    h0 = torch.randn(8, 64)
    h1 = ode_block(h0)

    assert h1.shape == h0.shape, f"Shape mismatch: {h1.shape} != {h0.shape}"
    print("Output shape:", h1.shape)
    print("NFE:", ode_block.nfe)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
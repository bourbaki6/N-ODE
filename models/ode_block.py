import torch
import torch.nn as nn
from .ode_solver import Euler, RK4, RK45
from .adjoint import AdjointODEBlock


class ODEBlock(nn.Module):

    def __init__(
        self,
        odefunc: nn.Module,
        solver: str = "rk4",
        num_steps: int = 10,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        t0: float = 0.0,
        t1: float = 1.0,
    ):
        super().__init__()
        self.odefunc = odefunc
        self.solver = solver
        self.num_steps = num_steps
        self.rtol = rtol
        self.atol = atol
        self.t0 = t0
        self.t1 = t1

        if solver == "adjoint":
            self._adjoint_block = AdjointODEBlock(odefunc, t0, t1)
       
        else:
            self._adjoint_block = None
 
        self.nfe_history = []

    
    def forward(self, h: torch.Tensor) -> torch.Tensor:

        self.odefunc.reset_nfe()

        if self.solver == "adjoint":
            h1 = self._adjoint_block(h)

        elif self.solver == "euler":
            h1 = Euler.euler_solve(
                self.odefunc, h,
                t0 = self.t0, t1 = self.t1,
                num_steps = self.num_steps,
            )

        elif self.solver == "rk4":

            h1=  RK4.rk4_solver(
                self.odefunc, h,
                t0 = self.t0, t1 = self.t1,
                num_steps = self.num_steps,
            )

        elif self.solver == "rk45":

            h1, rk45_nfe = RK45.rk45_solve(
                self.odefunc, h,
                t0 = self.t0, t1 = self.t1,
                rtol = self.rtol, atol = self.atol,
            )

            self.odefunc.nfe = rk45_nfe

        else:

            raise ValueError(
                f"Unknown solver '{self.solver}'. "
                f"Choose from: 'euler', 'rk4', 'rk45', 'adjoint'."
            )
        self.nfe_history.append(self.odefunc.nfe)

        return h1
    
    @property
    def nfe(self) -> int:

        return self.odefunc.nfe
        
    def set_solver(self, solver: str, num_steps: int = None):

        self.solver = solver

        if num_steps is not None:

            self.num_steps = num_steps

        if solver == "adjoint" and self._adjoint_block is None:
            self._adjoint_block = AdjointODEBlock(self.odefunc, self.t0, self.t1)

    def clear_nfe_history(self):

        self.nfe_history = []

        
if __name__ == "__main__":

    print("\n Done")



 
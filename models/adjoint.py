#---adjoint sensitivity method is the core 
#   contribution of teh Chen paper---#

import torch
import torch.nn as nn
from torch.autograd import Function
from .ode_solver import RK4


class AdjointODEFunc(Function):

    @staticmethod
    def forward(ctx, h0, t0, t1, func, *params):
        
        with torch.no_grad():
            h1 = RK4.rk4_solver(func, h0, t0 = t0.item(), t1 = t1.item(), num_steps = 10)

        ctx.save_for_backward(h0, t0, t1, h1)
        ctx.func = func
        ctx.n_params = len(params)

        return h1

    @staticmethod
    def backward(ctx, dL_dh1):
        
        h0, t0, t1, h1 = ctx.saved_tensors
        func = ctx.func
        n_params = ctx.n_params

        adj_h = dL_dh1.clone()

        adj_params = tuple(torch.zeros_like(p) for p in func.parameters())

        def augmented_dynamics(t, aug_state):
            
            h_aug = aug_state[0]
            a_aug = aug_state[1]

            h_aug = h_aug.detach().requires_grad_(True)

            with torch.enable_grad():
                f_val = func(t, h_aug)   

            vjp_inputs = (h_aug,) + tuple(func.parameters())
            grads = torch.autograd.grad(
                outputs = f_val,
                inputs = vjp_inputs,
                grad_outputs = a_aug,          
                allow_unused = True,
                retain_graph = False,
            )

            dh_dt = f_val.detach()         
            da_dt = -grads[0] if grads[0] is not None else torch.zeros_like(h_aug)     
            dp_dt = tuple(-g if g is not None else torch.zeros_like(p)
                           for g, p in zip(grads[1:], func.parameters()))

            return (dh_dt, da_dt) + dp_dt

        # Pack initial augmented state: (h1, a(1), zero_param_grads)
        aug_state = (h1.detach(), adj_h) + adj_params

        num_steps = 10
        dt = (t1.item() - t0.item()) / num_steps

        for i in range(num_steps - 1, -1, -1):
            t_cur = torch.tensor(
                t0.item() + i * dt,
                dtype=h0.dtype, device=h0.device
            )
            # Compute derivatives of augmented state at this time
            aug_derivs = augmented_dynamics(t_cur, aug_state)

            # Euler step BACKWARDS (subtract the forward derivative)
            aug_state = tuple(
                s - d * dt
                for s, d in zip(aug_state, aug_derivs)
            )

        # Unpack: aug_state[0] = h(0) (reconstructed), aug_state[1] = a(0) = dL/dh0
        dL_dh0 = aug_state[1]
        dL_dparams = aug_state[2:]

        return (dL_dh0, None, None, None) + dL_dparams


class AdjointODEBlock(nn.Module):

    def __init__(self, odefunc: nn.Module, t0: float = 0.0, t1: float = 1.0):
        super().__init__()
        self.odefunc = odefunc
        self.register_buffer('t0', torch.tensor(t0))
        self.register_buffer('t1', torch.tensor(t1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        
        self.odefunc.reset_nfe()

        # Collect current parameters for the custom Function
        params = tuple(self.odefunc.parameters())

        # Call the adjoint autograd Function
        h1 = AdjointODEFunc.apply(h, self.t0, self.t1, self.odefunc, *params)

        return h1

    @property
    def nfe(self):
        return self.odefunc.nfe
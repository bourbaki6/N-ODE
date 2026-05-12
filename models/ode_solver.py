#---Numerical ODE solvers: Euler, RK4, RK45 aka Dormand Price---#

import torch
from typing import Callable

from .odefunc import ODEFunction

class Euler:
    #---Only for 1st order ODE. Fixed step size---#
    @staticmethod
    def euler_solve(func: Callable, h0: torch.Tensor, t0: float = 0.0, t1: float = 1.0, num_steps: int= 10) -> torch.Tensor:
        h = h0
        dt = (t1 - t0) /num_steps
        dt_tensor = torch.tensor(dt, dtype = h0.dtype, device = h0.device)
        t = torch.tensor(t0, dtype = h0.dtype, device = h0.device)

        for _ in range(num_steps):
            #--- Euler update---#
            dh =  func(t, h)
            h = h + dh * dt_tensor
            t = t + dt_tensor

        return h
    
class RK4:

    @staticmethod
    def rk4_solver(func: Callable, h0: torch.Tensor, t0: float = 0.0, t1: float = 1.0, num_steps: int = 10) -> torch.Tensor:
        
        dt = (t1 - t0) /num_steps
        h = h0
        t = torch.tensor(t0, dtype = h0.dtype, device = h0.device) 
        dt_t = torch.tensor(dt, dtype = h0.dtype, device = h0.device)
        half_dt = dt_t * 0.5

        for _ in range(num_steps):
            #---slope estmates at current step---#
            k1 = func(t, h)
            k2 = func(t + half_dt, h + k1 * half_dt)
            k3 = func(t + half_dt, h + k2 * half_dt)
            k4 = func(t + dt_t, h + k3 * dt_t)

            #---weighted avg---#
            h = h + (dt_t / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            t = t + dt_t

        return h
    
#---solving using RK45 adaptive step method---#
class RK45:

    @staticmethod
    def rk45_solve(func: Callable, h0: torch.Tensor,
                   t0: float = 0.0, t1: float = 1.0,
                   rtol: float = 1e-3,atol: float = 1e-4,
                   min_dt: float = 1e-5, max_dt: float = 1.0,
                   ) -> tuple[torch.Tensor, int]:
        #---butcher tableau nodes---#
        c2, c3, c4, c5 = 1/5, 3/10, 4/5, 8/9

        #---Stage Coeff---#
        a21 = 1/5
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, - 56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

        #---5th order sol wts.(b*) used as the actual update---#
        b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

        #---4th order sol wts. (e) - only used for error estimate ie (diff between 4th and 5th order)---#
        e1 = 71/57600
        e3 = -71/16695
        e4 = 71/1920
        e5 = -17253/339200
        e6 = 22/525
        e7 = -1/40

        dtype, device = h0.dtype, h0.device
        t = t0
        h = h0
        dt = min(0.1, t1 - t0)
        nfe = 0

        while t < t1 - 1e-8:

            dt = min(dt, t1 - t)
            t_tensor = torch.tensor(t, dtype = dtype, device = device)
           
            k1 = func(t_tensor, h)
            k2 = func(t_tensor + c2*dt, h + dt*(a21 * k1))
            k3 = func(t_tensor + c3*dt, h + dt*(a31*k1 + a32*k2))
            k4 = func(t_tensor + c4*dt, h + dt*(a41*k1 + a42*k2 + a43*k3))
            k5 = func(t_tensor + c5*dt, h + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
            k6 = func(t_tensor + dt, h + dt*(a61*k1 + a62* k2 + a63*k3 + a64*k4 + a65*k5))
            nfe += 6

            #---5th solver to use for h_next---#
            h_next = h + dt * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)

            #---7th eval for error estimate---#
            k7 = func(t_tensor + dt, h_next)
            nfe += 1
            
            err_vec = dt * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 *k6 + e7 * k7)
            
            tol = atol + rtol * torch.max(h.abs(), h_next.abs())
            
            err_norm = (err_vec / tol).pow(2).mean().sqrt().item()

            if err_norm <= 1.0:

                t = t + dt
                h = h_next
            
            if err_norm == 0.0:
                factor = 5.0
 
            else:
                factor = min(5.0, max(0.2, 0.9 * (1.0 / err_norm) ** 0.2))

            dt = min(max(dt * factor, min_dt), max_dt)

        return h, nfe


if __name__ == "__main__":
    
    f = ODEFunction(64)
    h = torch.randn(4, 64)
    t = torch.tensor(0.5)
    print(f(t, h).shape)  
    print("\n Done")




 









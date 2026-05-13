# Neural Ordinary Differential Equation Solver - Reproduction & Analysis
 
A from-scratch implementation of **Neural Ordinary Differential Equations** (Chen et al., NeurIPS 2018) for image classification, with a full solver suite, custom adjoint backpropagation, and comparative analysis against a ResNet baseline.

## Overview
 
This project reproduces the core ideas of the Neural ODE paper: replacing discrete residual blocks with a continuous-time ODE defined over a learned vector field. A single `ODEFunction` is integrated from `t=0` to `t=1` using a choice of numerical solvers, and the whole system is trained end-to-end via either standard backprop-through-solver or the adjoint sensitivity method.
 
Three models are trained and compared on MNIST:
 
| Model | Test Accuracy | Parameters | NFE / forward pass |
|---|---|---|---|
| Neural ODE (Euler) | 97.84% | ~50K | 10 |
| Neural ODE (RK4) | 97.63% | ~50K | 40 |
| ResNet-6 (baseline) | 97.95% | ~120K | — |
 
The ODE models match ResNet-6 accuracy while using ~58% fewer parameters.
 

## Architecture
 
```
Input (784)
    │
    ▼
Linear → GroupNorm → Tanh          # input_proj: project to hidden_dim = 64
    │
    ▼
ODEBlock: integrate dh/dt = f(h,t) from t = 0 to t = 1
    │   └── ODEFunction: Linear(65→64) -> Tanh -> Linear(64 -> 64)
    │         (time t concatenated to h at each step)
    ▼
GroupNorm → Linear(64 -> 10)          # output_proj
    │
    ▼
log_softmax → NLLLoss
```

## Reference
 
> Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018).
> **Neural Ordinary Differential Equations.** NeurIPS 2018.
> https://arxiv.org/abs/1806.07366

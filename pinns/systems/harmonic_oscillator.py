from pinns.systems.system import System
import deepxde as dde
import torch
import numpy as np
from typing import cast


class HarmonicOscillator0D(System):
    """Class modeling the harmonic oscillator in 0+1 dimensions. The ODE is:
    $$
        \\frac{\\partial^2 y}{\\partial t^2} + \\xi_0 * \\omega_0 * \\frac{\\partial y}{\\partial t} + \\frac{\\omega_0^2}{4} y = 0
    $$
    and with initial conditions $y(t = 0) = y_0, y^\\prime(t = 0) = y_1$.


    Args:
        omega0 (float): Intrinsic frequency of the system.
        xi0 (float): Fluid friction term.
    """

    __slots__ = ("xi0", "omega0", "xi", "omega", "y0", "y1")

    def __init__(self, xi0: float, omega0: float, y0: float, y1: float):
        # ODE parameters
        self.xi0 = xi0
        self.omega0 = omega0
        self.xi = xi0 * omega0 / 2
        self.omega = self.omega0 * np.sqrt(np.abs(np.power(self.xi0, 2) - 1)) / 2

        # Initial value condition
        self.y0 = y0
        self.y1 = y1

        # Define the geometry

    def equation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dy_dt = cast(torch.Tensor, dde.grad.jacobian(y, x, i=0, j=0))
        dy2_dt2 = cast(torch.Tensor, dde.grad.hessian(y, x))
        ode = dy2_dt2 + self.xi0 * self.omega0 * dy_dt + (self.omega0**2 / 4) * y
        return ode

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.has_exact_sol:
            return None

        # Damping term
        exp_term = torch.exp(-self.xi * x)
        # Critical solution
        if self.xi0 == 1:
            c0 = self.y0
            c1 = self.y1 + self.xi * self.y0
            return exp_term * (c0 + c1 * x)
        else:
            c0 = self.y0
            c1 = (self.y1 + self.xi * self.y0) / self.omega
            # Underdamped case
            if self.xi0 < 1:
                return exp_term * (c0 * torch.cos(self.omega * x) + c1 * torch.sin(self.omega * x))
            # Overdamped case
            else:
                return exp_term * (c0 * torch.cosh(self.omega * x) + c1 * torch.sinh(self.omega * x))

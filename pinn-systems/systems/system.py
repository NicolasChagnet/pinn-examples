import torch
from abc import ABC, abstractmethod


class System(ABC):
    """Generic class detailing the interface of a PDE system."""

    __slots__ = ("has_exact_sol",)

    @abstractmethod
    def equation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Given the input and the model output, returns the equation to be solved."""
        pass

    @abstractmethod
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor | None:
        """If provided, returns the exact solution for these inputs."""
        pass

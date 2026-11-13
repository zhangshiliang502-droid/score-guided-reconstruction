from __future__ import annotations

import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, omega0: float = 20.0, is_first: bool = False) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.omega0 = float(omega0)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(int(in_dim), int(out_dim))
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_dim
            else:
                bound = math.sqrt(6.0 / self.in_dim) / self.omega0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class SIRENImplicitSurface(nn.Module):
    """SIREN implicit hypersurface Phi_theta: R^d -> R."""

    def __init__(
        self,
        hidden_dim: int = 128,
        hidden_layers: int = 4,
        *,
        omega0: float = 20.0,
        input_dim: int = 3,
    ) -> None:
        super().__init__()
        if int(hidden_layers) < 1:
            raise ValueError("hidden_layers must be >= 1")
        layers: list[nn.Module] = [
            SineLayer(int(input_dim), int(hidden_dim), omega0=float(omega0), is_first=True)
        ]
        for _ in range(int(hidden_layers) - 1):
            layers.append(
                SineLayer(
                    int(hidden_dim),
                    int(hidden_dim),
                    omega0=float(omega0),
                    is_first=False,
                )
            )
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.omega0 = float(omega0)
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(int(hidden_dim), 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / int(hidden_dim)) / float(omega0)
            self.final.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.net(x)).squeeze(-1)

    def phi_and_grad(self, x: torch.Tensor, *, create_graph: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        phi = self.forward(x)
        grad = torch.autograd.grad(
            phi.sum(),
            x,
            create_graph=create_graph,
            retain_graph=create_graph,
            only_inputs=True,
        )[0]
        return phi, grad

    def phi_and_grad_analytic(
        self,
        x: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Phi and its exact input gradient by the SIREN chain rule."""

        context = torch.enable_grad() if create_graph else torch.no_grad()
        with context:
            hidden = x
            derivative_factors: list[torch.Tensor] = []
            for layer in self.net:
                if not isinstance(layer, SineLayer):
                    raise TypeError(
                        "phi_and_grad_analytic requires every hidden module to be a SineLayer"
                    )
                preactivation = layer.omega0 * layer.linear(hidden)
                derivative_factors.append(layer.omega0 * torch.cos(preactivation))
                hidden = torch.sin(preactivation)

            phi = self.final(hidden).squeeze(-1)
            gradient = self.final.weight.expand(x.shape[0], -1)
            for layer, derivative_factor in zip(
                reversed(self.net),
                reversed(derivative_factors),
            ):
                gradient = torch.matmul(
                    gradient * derivative_factor,
                    layer.linear.weight,
                )
        return phi, gradient

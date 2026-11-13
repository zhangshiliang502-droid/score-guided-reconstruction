"""Neural models shared by the generic high-dimensional pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SineLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        omega0: float = 20.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.omega0 = float(omega0)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(int(in_dim), int(out_dim))
        with torch.no_grad():
            bound = (
                1.0 / self.in_dim
                if self.is_first
                else math.sqrt(6.0 / self.in_dim) / self.omega0
            )
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class SIRENImplicitSurface(nn.Module):
    """SIREN implicit hypersurface Phi: R^D -> R."""

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
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.omega0 = float(omega0)
        self.input_dim = int(input_dim)
        layers: list[nn.Module] = [
            SineLayer(
                self.input_dim,
                self.hidden_dim,
                omega0=self.omega0,
                is_first=True,
            )
        ]
        for _ in range(self.hidden_layers - 1):
            layers.append(
                SineLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    omega0=self.omega0,
                )
            )
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(self.hidden_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / self.hidden_dim) / self.omega0
            self.final.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.net(x)).squeeze(-1)

    def phi_and_grad(
        self,
        x: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = x if x.requires_grad else x.detach().requires_grad_(True)
        phi = self.forward(query)
        gradient = torch.autograd.grad(
            phi.sum(),
            query,
            create_graph=bool(create_graph),
            retain_graph=bool(create_graph),
            only_inputs=True,
        )[0]
        return phi, gradient


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequency = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half,
                device=time.device,
                dtype=time.dtype,
            )
        )
        angle = time.reshape(-1, 1) * frequency[None, :] * (2.0 * math.pi)
        embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)
        return F.pad(embedding, (0, self.dimension % 2))


class EnergyMLP(nn.Module):
    """Scalar scaled potential U=sigma*E used for conservative score learning."""

    def __init__(
        self,
        data_dim: int,
        time_dim: int = 64,
        hidden_dim: int = 256,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.data_dim = int(data_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(int(time_dim)),
            nn.Linear(int(time_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.in_proj = nn.Linear(self.data_dim, int(hidden_dim))
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                    nn.SiLU(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.out_proj = nn.Linear(int(hidden_dim), 1, bias=False)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"x must have shape [B,{self.data_dim}]")
        time_feature = self.time_embed(time.reshape(-1))
        hidden = F.silu(self.in_proj(x))
        for block in self.blocks:
            hidden = F.silu(hidden + block(torch.cat((hidden, time_feature), dim=1)))
        return self.out_proj(hidden).squeeze(1)


def energy_gradient(
    model: nn.Module,
    x: torch.Tensor,
    time: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = x if x.requires_grad else x.detach().requires_grad_(True)
    energy = model(query, time)
    gradient = torch.autograd.grad(
        energy.sum(),
        query,
        create_graph=bool(create_graph),
    )[0]
    return energy, gradient, -gradient

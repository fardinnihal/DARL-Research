"""He and Randomized Asymmetric Initialization (RAI) for the A1 network.

He initialization (He et al. 2015; as used by Lu et al.)
    Every Linear layer: W ~ N(0, 2 / fan_in), b = 0.
    Zero bias is a labelled assumption: Lu et al. say only "He initialization";
    He et al. initialise biases to 0, and Lu use b = 0 for RAI's first layer.

RAI (Lu et al., Section 4.1 and Appendix G code)
    Layer 1:   He, W ~ N(0, 2 / d_in), b = 0.
    Layer l>=2 (including the output layer), for each output unit j:
        V_j = [W_j, b_j] in R^(fan_in + 1),
        V_j ~ N(0, sigma^2 I) with sigma = 0.6007 / sqrt(fan_in)   (Appendix G)
        pick k_j uniformly from {0, ..., fan_in}   (k_j = fan_in is the bias)
        V_j[k_j] ~ Beta(2, 1)
    Lu's Section 4.3 states sigma_w ~= 0.6007 without the 1/sqrt(fan_in) factor;
    the Appendix G code uses 0.6007 / fan_in**0.5. We follow the code, which is
    what produced their reported numbers, and report the discrepancy.
    The constant 0.6007 is the code's literal; ``RAI_SIGMA_FORMULA`` is the
    closed form from Section 4.3, kept to show they agree to 4 decimals.

    Beta(2, 1) has CDF F(u) = u^2 on [0, 1], so sqrt(U) with U ~ Uniform[0, 1)
    is an exact Beta(2, 1) sample. This lets one seeded torch.Generator drive
    every draw.

Seeds
-----
A run seed s is expanded with numpy.random.SeedSequence into independent
streams ("init", "order", ...). The init stream depends only on s, never on
the activation, so ReLU, LeakyReLU and DARL arms with the same seed start from
bitwise identical weight tensors. The draw order is fixed and documented in
each function, so the tensors are reproducible on any CPU.
Initialisation never touches the global torch, NumPy or Python RNG.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

__all__ = [
    "INIT_SCHEMES",
    "RAI_SIGMA",
    "RAI_SIGMA_FORMULA",
    "SEED_STREAMS",
    "derive_seeds",
    "make_generator",
    "he_init_",
    "rai_init_",
    "initialize_",
]

INIT_SCHEMES = ("he", "rai")

RAI_SIGMA = 0.6007  # Appendix G literal
RAI_SIGMA_FORMULA = -2.0 * math.sqrt(2.0) / (3.0 * math.sqrt(math.pi)) + math.sqrt(1.0 + 8.0 / (9.0 * math.pi))

# Named, ordered streams spawned from a run seed. Append new names at the END
# only: SeedSequence.spawn assigns children by position.
SEED_STREAMS = ("init", "order")


def derive_seeds(run_seed: int) -> Dict[str, int]:
    """Expand a run seed into one independent 63-bit seed per named stream."""
    if isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed < 0:
        raise ValueError(f"run_seed must be a non-negative int, got {run_seed!r}")
    children = np.random.SeedSequence(run_seed).spawn(len(SEED_STREAMS))
    return {
        name: int(child.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
        for name, child in zip(SEED_STREAMS, children)
    }


def make_generator(seed: int) -> torch.Generator:
    """A private CPU generator; the global torch RNG is not touched."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _linears(model: nn.Module) -> List[nn.Linear]:
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not layers:
        raise ValueError("model has no nn.Linear layers")
    return layers


@torch.no_grad()
def _he_layer_(layer: nn.Linear, g: torch.Generator) -> None:
    fan_in = layer.in_features
    std = math.sqrt(2.0 / fan_in)
    w = torch.randn(layer.weight.shape, generator=g, dtype=torch.float64) * std
    layer.weight.copy_(w.to(layer.weight.dtype))
    if layer.bias is not None:
        layer.bias.zero_()


@torch.no_grad()
def he_init_(model: nn.Module, generator: torch.Generator) -> None:
    """He normal init of every Linear layer, in module order. Draw order:
    for each layer, one randn of the weight shape (row-major)."""
    for layer in _linears(model):
        _he_layer_(layer, generator)


@torch.no_grad()
def _rai_layer_(layer: nn.Linear, g: torch.Generator) -> torch.Tensor:
    if layer.bias is None:
        raise ValueError("RAI needs layers with a bias (the bias is part of V_j)")
    fan_out, fan_in = layer.out_features, layer.in_features
    sigma = RAI_SIGMA / math.sqrt(fan_in)
    # Draw order: normals for V (fan_out x (fan_in+1)), then k (fan_out), then U (fan_out).
    v = torch.randn((fan_out, fan_in + 1), generator=g, dtype=torch.float64) * sigma
    k = torch.randint(0, fan_in + 1, (fan_out,), generator=g)
    u = torch.rand((fan_out,), generator=g, dtype=torch.float64)
    v[torch.arange(fan_out), k] = u.sqrt()  # Beta(2, 1) by inverse CDF
    layer.weight.copy_(v[:, :fan_in].to(layer.weight.dtype))
    layer.bias.copy_(v[:, fan_in].to(layer.bias.dtype))
    return k


@torch.no_grad()
def rai_init_(model: nn.Module, generator: torch.Generator) -> List[Optional[torch.Tensor]]:
    """RAI (Lu et al.). Returns the chosen index k_j per layer (None for layer 1),
    so tests and audits can locate the Beta(2, 1) entries."""
    layers = _linears(model)
    chosen: List[Optional[torch.Tensor]] = [None]
    _he_layer_(layers[0], generator)
    for layer in layers[1:]:
        chosen.append(_rai_layer_(layer, generator))
    return chosen


def initialize_(model: nn.Module, scheme: str, run_seed: int) -> Dict[str, int]:
    """Initialise ``model`` in place from a run seed. Returns the derived seeds."""
    if scheme not in INIT_SCHEMES:
        raise ValueError(f"unknown init scheme {scheme!r}; choose from {INIT_SCHEMES}")
    seeds = derive_seeds(run_seed)
    g = make_generator(seeds["init"])
    if scheme == "he":
        he_init_(model, g)
    else:
        rai_init_(model, g)
    return seeds

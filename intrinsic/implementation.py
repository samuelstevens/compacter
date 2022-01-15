import random
from typing import List, NamedTuple, Tuple

import numpy as np
import torch

if torch.cuda.is_available():
    from .fwh_cuda import fast_walsh_hadamard_transform  # type: ignore
else:
    from .fwh import fast_walsh_hadamard_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HiddenParam(NamedTuple):
    name: str
    module: torch.nn.Module
    module_name: str
    shape: torch.Size
    numel: int


def make_hidden_params(module) -> Tuple[List[HiddenParam], torch.Tensor]:
    hidden_params = []
    theta_0s = []

    # Iterate over layers in the module
    for name, param in sorted(list(module.named_parameters())):
        # If param does not require update, skip it because we are not tuning it.
        if not param.requires_grad:
            continue

        # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
        theta_0s.append(param.detach().requires_grad_(False))

        base, localname = module, name
        while "." in localname:
            prefix, localname = localname.split(".", 1)
            base = getattr(base, prefix)

        numel = int(np.prod(param.shape))
        hidden_params.append(HiddenParam(name, base, localname, param.shape, numel))

    flat_theta_0s = []
    for theta_0 in theta_0s:
        if len(theta_0.shape) > 1:
            theta_0 = theta_0.flatten().squeeze()
        flat_theta_0s.append(theta_0)

    # Stores the initial value: \theta_{0}^{D}
    theta_0 = torch.cat(flat_theta_0s)

    return hidden_params, theta_0


class FastWalshHadamard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(
            torch.tensor([1 / np.sqrt(float(input.size(0)))]).to(input)
        )
        return fast_walsh_hadamard_transform(input.float(), False)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return input * fast_walsh_hadamard_transform(
            grad_output.clone().float(), False
        ).to(grad_output)

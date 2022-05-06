# The codes are from Armen Aghajanyan from facebook, from paper
# Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
# https://arxiv.org/abs/2012.13255

import logging

import numpy as np
import torch

from . import implementation


class FastfoodTransform(torch.nn.Module):
    logger = logging.getLogger("FastfoodTransform")

    def __init__(self, d, D):
        super().__init__()
        self.d = d
        self.D = D

        # smallest integer that is larger than log base 2 of dimension
        ll = int(np.ceil(np.log2(self.D)))
        self.LL = 2**ll

        # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
        BB = torch.randint(0, 2, size=(self.LL,))
        BB = (BB * 2 - 1).type(torch.FloatTensor)
        BB.requires_grad = False
        self.register_buffer("BB", BB)

        # Random permutation matrix
        Pi = torch.randperm(self.LL)
        Pi.requires_grad = False
        self.register_buffer("Pi", Pi)

        # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
        GG = torch.randn(self.LL)
        GG.requires_grad = False
        self.register_buffer("GG", GG)

        # single divisor to normalize transform
        divisor = torch.sqrt(self.LL * torch.sum(torch.pow(self.GG, 2)))
        self.register_buffer("divisor", divisor)

        self.walsh_hadamard = None

    def forward(self, x):
        """
        Fastfood transform
        :param x: array of dd dimension
        :return:
        """
        assert x.shape == (self.d,)

        # Pad x if needed
        d_pad = torch.nn.functional.pad(
            x, pad=(0, self.LL - self.d), value=0, mode="constant"
        )

        # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
        mul_1 = self.BB * d_pad

        # HGPi(HBX)
        mul_2 = implementation.FastWalshHadamard.apply(mul_1)

        # HG(PiHBX)
        mul_3 = mul_2[self.Pi]

        # H(GPiHBX)
        mul_4 = mul_3 * self.GG

        # (HGPiHBX)
        mul_5 = implementation.FastWalshHadamard.apply(mul_4)

        return torch.div(
            mul_5[: self.D], self.divisor * np.sqrt(float(self.D) / self.LL)
        )


class IntrinsicDimension(torch.nn.Module):
    logger = logging.getLogger("intrinsic.IntrinsicDimension")

    def __init__(
        self, module: torch.nn.Module, int_dim: int, said: bool, fastfood_seed: int
    ):
        super().__init__()

        # Hide the module from inspection by get_parameters()
        self.m = [module]

        self.device = (
            module.device if hasattr(module, "device") else torch.device("cpu")
        )

        self.use_said = said

        self.hidden_params, self.theta_0 = implementation.make_hidden_params(module)

        for hidden_param in self.hidden_params:
            delattr(hidden_param.module, hidden_param.module_name)

        self.fastfood_seed = fastfood_seed
        implementation.set_seed(self.fastfood_seed)

        D = self.theta_0.shape[0]
        self.fastfood = FastfoodTransform(int_dim, D)

        self.said_size = len(self.hidden_params)
        if self.use_said:
            assert int_dim > self.said_size
            int_dim -= self.said_size + 1

        self.d = int_dim
        self.intrinsic_vector = torch.nn.Parameter(torch.zeros((int_dim)))

        if self.use_said:
            self.said_parameter = torch.nn.Parameter(torch.ones((self.said_size)))

        self.logger.info(
            f"Initialized Fastfood wrapper around {module.__class__.__name__}."
        )

    def set_module_weights(self):
        updated = self.fastfood(self.intrinsic_vector)

        start, end = 0, 0
        for i, hidden_param in enumerate(self.hidden_params):
            start = end
            end = start + hidden_param.numel
            theta_0 = self.theta_0[start:end].view(hidden_param.shape)
            update = updated[start:end].view(hidden_param.shape)
            if self.use_said:
                update *= self.said_parameter[i]

            setattr(hidden_param.module, hidden_param.module_name, theta_0 + update)

    def forward(self, *args, **kwargs):
        # Uses the intrinsic dimension vector to update the underlying model weights.
        self.set_module_weights()

        # Does normal forward pass of underlying model.
        return self.hidden(*args, **kwargs)

    @property
    def hidden(self):
        return self.m[0]

    def __getattr__(self, name):
        """
        Somehow we need to call super().__getattr__ to check for model parameters - self._parameters in self.register_parameter
        """
        if hasattr(self.hidden, name):
            return getattr(self.hidden, name)

        return super().__getattr__(name)

    def to(self, device, non_blocking=False):
        self.logger.debug(
            "Before moving [max memory allocated: %.3f]",
            torch.cuda.max_memory_allocated(),
        )

        super().to(device)  # moves theta_d
        self.logger.debug(
            "After super().to(device) [max memory allocated: %.3f]",
            torch.cuda.max_memory_allocated(),
        )

        self.theta_0 = implementation.send_to_device(self.theta_0, device)
        self.logger.debug(
            "After moving theta_0 [max memory allocated: %.3f]",
            torch.cuda.max_memory_allocated(),
        )

        self.set_module_weights()  # moves base model
        self.logger.debug(
            "After updating base model weights [max memory allocated: %.3f]",
            torch.cuda.max_memory_allocated(),
        )
        self.hidden.to(device)
        self.logger.debug(
            "After moving base model [max memory allocated: %.3f]",
            torch.cuda.max_memory_allocated(),
        )

        self.device = device

        with torch.no_grad():
            torch.cuda.empty_cache()

        return self


__all__ = ["IntrinsicDimension", "FastfoodTransform"]

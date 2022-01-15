import torch
import transformers

from .. import implementation


class NeuralNetwork(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for in_size, out_size in zip(layers, layers[1:]):
            self.layers.append(torch.nn.Linear(in_size, out_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.relu(x)

        return x


def test_make_hidden_params_single_layer():
    model = NeuralNetwork(layers=[10, 1])

    hidden_params, theta_0 = implementation.make_hidden_params(model)

    assert theta_0.shape == (11,)  # 10 weights + 1 bias
    assert len(hidden_params) == 2
    assert [hp.name for hp in hidden_params] == [
        name for name, param in sorted(model.named_parameters())
    ]


def test_make_hidden_params_three_layers():
    model = NeuralNetwork(layers=[256, 128, 32, 10])

    hidden_params, theta_0 = implementation.make_hidden_params(model)

    assert theta_0.shape == (257 * 128 + 129 * 32 + 33 * 10,)
    assert len(hidden_params) == 6
    assert [hp.name for hp in hidden_params] == [
        name for name, param in sorted(model.named_parameters())
    ]


def test_make_hidden_params_gpt2():
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    hidden_params, theta_0 = implementation.make_hidden_params(model)

    assert [hp.name for hp in hidden_params] == [
        name for name, param in sorted(model.named_parameters())
    ]

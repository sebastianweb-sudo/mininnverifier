# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
import numpy as np
from scipy import special

from . import core


class Array(core.Value):
    def __init__(self, array_like):
        self.array = np.asarray(array_like, dtype=np.float64)
        super().__init__(EvalInterpreter(), self.array.shape)

    def item(self):
        return self.array.item()

    def __repr__(self):
        data_str = str(self.array).replace("\n", "\n" + " " * len("Array("))
        return f"Array({data_str})"


def full(shape, fill_value):
    return Array(np.full(shape, fill_value, dtype=np.float64))


def zeros(shape):
    return full(shape, 0.0)


def ones(shape):
    return full(shape, 1.0)


class EvalInterpreter(core.Interpreter[Array]):
    def __init__(self):
        super().__init__(0)

    def wrap(self, value):
        if not isinstance(value, core.Value):
            return Array(value)
        elif not isinstance(value, Array):
            raise ValueError("EvalInterpreter must be the bottom interpreter")
        return value

    def process(self, primitive, values: list[Array], options: dict):
        np_vals = [v.array for v in values]
        np_out = eval_rules[primitive](*np_vals, **options)
        return Array(np_out)


def np_dot(x, y):  # np.dot doesn't broadcast
    if y.ndim <= 1:
        return np.dot(x, y)
    return np.einsum("...j,...jk", x, y)

def np_pad(x, config, axes, value):
    ℓ, r, m = config
    pad_width = [(0, 0)] * x.ndim
    for ax in axes:
        pad_width[ax] = (ℓ, r)
    y = np.pad(x, pad_width, constant_values=value)
    if m > 0:
        shape = list(y.shape)
        for ax in axes:
            new_shape = shape[ax] + (shape[ax] - 1) * m
            expanded = np.full(new_shape, value, dtype=np.float64)
            idx = [slice(None)] * y.ndim
            idx[ax] = slice(ℓ, new_shape - r, m + 1)
            expanded[tuple(idx)] = y
            y = expanded
    return y

def np_conv2d(x, K, stride=1):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = K.shape

    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    # 1) Extrahiere alle Patches aus x (im2col)
    patches = np.lib.stride_tricks.as_strided(
        x,
        shape=(N, C_in, H_out, W_out, kH, kW),
        strides=(
            x.strides[0],
            x.strides[1],
            x.strides[2] * stride,
            x.strides[3] * stride,
            x.strides[2],
            x.strides[3],
        ),
        writeable=False,
    )

    y = np.einsum("nchwij, Ccij -> nChw", patches, K)

    return y





eval_rules = {
    core.expand_dims: lambda x, axes: np.expand_dims(x, axes),
    core.moveaxis: np.moveaxis,
    core.reshape: lambda x, new_shape: np.reshape(x, new_shape),
    core.neg: lambda x: -x,
    core.add: lambda x, y: x + y,
    core.reduce_sum: lambda x, axes: x.sum(axes),
    core.dot: np_dot,
    core.mul: lambda x, y: x * y,
    core.reciprocal: lambda x: 1 / x,
    core.relu: lambda x: np.maximum(x, 0.0),
    core.leaky_relu :lambda x, negative_slope: np.maximum(x, 0.0) + negative_slope * np.minimum(x, 0.0), 
    core.elu :lambda x, alpha: np.where(x > 0, x, alpha * (np.exp(x) - 1)),
    core.gelu: lambda x : x * 1/2 * (1 + special.erf(x/np.sqrt(2))),
    core.square: np.square,
    core.sqrt: np.sqrt,
    core.exp: np.exp,
    core.log: np.log,
    core.flip: lambda x, axes: np.flip(x, axis=axes),
    core.where: np.where,
    core.pad: np_pad,
    core.conv: np_conv2d,

}

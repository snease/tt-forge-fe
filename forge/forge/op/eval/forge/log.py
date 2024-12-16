# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import torch.nn.functional
from loguru import logger

from forge.op.eval.common import calculate_tile_size

from ....forgeglobal import TILE_DIM
from ....tensor import forge_dataformat_to_pytorch_dtype
from ..common import to_torch_operands
from ..interface import PyEltwiseUnaryOp
from ..lforge.log import Log as ForgeLog
from .reciprocal import Reciprocal


class Log(PyEltwiseUnaryOp):
    @classmethod
    def create(cls):
        self = cls("log")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Log should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = torch.log(tensors[0] + 1e-10)

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Log should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert len(inputs) == 1, "Log should have one input"
        assert operand == 0, "Invalid operand index"
        recip = ac.op(Reciprocal.create(), (inputs[0],))
        return ac.op("multiply", (recip, grad))

    def lower(self, lc, tensors, outputs):
        assert len(tensors) == 1, "Log should  have one input"

        if bool(int(os.environ.get("FORGE_ENABLE_TINY_TILE", "0"))):
            node_shape = list(tensors[0].shape)
            tile_height = calculate_tile_size(node_shape[-2])
            tile_width = calculate_tile_size(node_shape[-1])
            vector = "" if tile_height == TILE_DIM else "r"
        else:
            vector = None
            tile_height, tile_width = TILE_DIM, TILE_DIM

        lc.op(
            ForgeLog.create(vector=vector),
            tensors,
            tile_height=tile_height,
            tile_width=tile_width,
        )

    def initial_flops_estimate(self, tensor_shapes):
        flops = 0
        output_shape = self.shape(tensor_shapes)[0]
        flops = np.prod(output_shape)

        return flops

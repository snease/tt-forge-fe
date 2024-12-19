# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Building PyTorch models


import torch

from typing import Type
from loguru import logger

from forge import PyTorchModule, ForgeModule

from .. import RandomizerGraph, ModelBuilder, StrUtils


class PyTorchModelBuilder(ModelBuilder):

    def build_model(self, graph: RandomizerGraph, GeneratedTestModel: Type[torch.nn.Module]) -> ForgeModule:
        pytorch_model = GeneratedTestModel()
        # module_name = f"gen_model_pytest_{StrUtils.test_id(graph)}"
        # pybuda_model = PyTorchModule(module_name, pytorch_model)
        return pytorch_model

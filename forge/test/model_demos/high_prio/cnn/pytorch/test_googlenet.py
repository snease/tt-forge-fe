# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from torchvision import models, transforms
from test.utils import download_model
import forge
from PIL import Image
from loguru import logger
import os


def test_googlenet_pytorch(test_device):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    # compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Create Forge module from PyTorch model
    # Two ways to load the same model
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    model = download_model(models.googlenet, pretrained=True)
    model.eval()

    class wrapper_forge(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.maxpool1 = model.maxpool1

        def forward(self, x):
            x = self.maxpool1(x)
            return x

    model = wrapper_forge(model)
    print("wrapped model", model)
    inputs = [torch.rand(1, 64, 112, 112)]

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_googlenet_d4")

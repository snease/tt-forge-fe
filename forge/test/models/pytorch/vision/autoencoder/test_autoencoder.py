# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
import os
import pytest
from test.models.pytorch.vision.autoencoder.utils.conv_autoencoder import ConvAE
from test.models.pytorch.vision.autoencoder.utils.linear_autoencoder import LinearAE
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_conv_ae_pytorch(test_device):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    model = ConvAE()

    # Define transform to normalize data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load sample from MNIST dataset
    dataset = load_dataset("mnist")
    sample = dataset["train"][0]["image"]
    sample_tensor = transform(sample).unsqueeze(0)

    compiled_model = forge.compile(model, sample_inputs=[sample_tensor], module_name="pt_conv_ae")


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="ValueError: Data mismatch (all_close)")
def test_linear_ae_pytorch(test_device):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()

    # Instantiate model
    # NOTE: The model has not been pre-trained or fine-tuned.
    # This is for demonstration purposes only.
    model = LinearAE()

    # Define transform to normalize data
    transform = transforms.Compose(
        [
            transforms.Resize((1, 784)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load sample from MNIST dataset
    dataset = load_dataset("mnist")
    sample = dataset["train"][0]["image"]
    sample_tensor = transform(sample).squeeze(0)

    # Sanity
    fw_out = model(sample_tensor)

    # Inference
    compiled_model = forge.compile(model, sample_inputs=[sample_tensor], module_name="pt_linear_ae")
    verify([sample_tensor], model, compiled_model)

# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import forge
import os
import forge
from test.utils import download_model
import torch
from PIL import Image
from torchvision import transforms
import urllib


def generate_model_wideresnet_imgcls_pytorch(test_device, variant):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["FORGE_RIBBON2"] = "1"

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", variant, pretrained=True)
    framework_model.eval()
    model_name = f"pt_{variant}"
    # tt_model = forge.PyTorchModule(model_name,framework_model)

    # STEP 3: Prepare input
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    img_tensor = input_tensor.unsqueeze(0)

    return framework_model, [img_tensor]


variants = ["wide_resnet50_2"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_wideresnet_pytorch(variant, test_device):
    (model, inputs,) = generate_model_wideresnet_imgcls_pytorch(
        test_device,
        variant,
    )

    print("inputs[0].shape", inputs[0].shape)

    class wrapper_forge(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.conv1 = model.conv1
            self.bn1 = model.bn1

        def forward(self, x):
            from loguru import logger

            logger.info("input shape={}", x.shape)
            x = self.conv1(x)
            logger.info("after conv1 shape={}", x.shape)
            x = self.bn1(x)
            return x

    wrapped_model = wrapper_forge(model)

    print("wrappered_model", wrapped_model)

    compiled_model = forge.compile(wrapped_model, sample_inputs=inputs, module_name="oct16_debug_1")
    co_out = compiled_model(*inputs)

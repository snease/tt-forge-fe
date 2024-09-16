import torch
import pytest
from torchvision.models.vision_transformer import vit_b_16
from transformers import ViTForImageClassification

import forge
from test.utils import download_model


def test_vit_inference_hf():
    # Load ViT base model
    framework_model = download_model(
        ViTForImageClassification.from_pretrained, "google/vit-base-patch16-224"
    )
    framework_model.eval()
    
    # Prepare input tensor
    input_tensor = torch.rand((1, 3, 224, 224))

    # Sanity run
    cpu_output = framework_model(input_tensor)

    # Compile the model
    compiled_model = forge.compile(framework_model, input_tensor)

    # Run inference
    output = compiled_model(input_tensor)
    print(output)


@pytest.mark.skip()
def test_vit_inference_tv():
    # Load ViT base model
    framework_model = vit_b_16(pretrained=True)
    framework_model.eval()
    
    # Prepare input tensor
    input_tensor = torch.rand((1, 3, 224, 224))

    # Sanity run
    cpu_output = framework_model(input_tensor)
    print(cpu_output.shape)

    # Compile the model
    compiled_model = forge.compile(framework_model, input_tensor)

    # Run inference
    output = compiled_model(input_tensor)
    print(output)

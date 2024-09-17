import torch
import pytest
from torchvision.models.vision_transformer import vit_b_16
from transformers import ViTForImageClassification

import forge
from test.utils import download_model


@pytest.mark.skip()
def test_vit_norm():
    # Load ViT base model
    framework_model = download_model(
        ViTForImageClassification.from_pretrained, "google/vit-base-patch16-224"
    )
    framework_model.eval()
    
    # Limit the model to the embeddings layer
    framework_model = framework_model.vit.encoder.layer[0].layernorm_after
    
    # Prepare input tensor
    input_tensor = torch.rand((1, 197, 768))

    # Sanity run
    cpu_output = framework_model(input_tensor)

    # Compile the model
    compiled_model = forge.compile(framework_model, input_tensor)

    # Run inference
    output = compiled_model(input_tensor)

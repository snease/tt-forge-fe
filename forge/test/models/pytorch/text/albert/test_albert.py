# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
from transformers import AlbertForMaskedLM, AlbertTokenizer, AlbertForTokenClassification


sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.parametrize("size", sizes, ids=sizes)
@pytest.mark.nightly
def test_albert_masked_lm_pytorch(size, variant, test_device):
    model_ckpt = f"albert-{size}-{variant}"

    # Load Albert tokenizer and model from HuggingFace
    tokenizer = download_model(AlbertTokenizer.from_pretrained, model_ckpt)
    model = download_model(AlbertForMaskedLM.from_pretrained, model_ckpt)

    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Load data sample
    sample_text = "The capital of France is [MASK]."

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    model(**input_tokens)

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]
    varaint_name = model_ckpt.replace("-", "_")
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=f"pt_{varaint_name}_masked_lm")


sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.parametrize("size", sizes, ids=sizes)
@pytest.mark.nightly
def test_albert_token_classification_pytorch(size, variant, test_device):

    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    # Variants: albert-base-v1, albert-large-v1, albert-xlarge-v1, albert-xxlarge-v1
    # albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2
    model_ckpt = f"albert-{size}-{variant}"

    # Load ALBERT tokenizer and model from HuggingFace
    tokenizer = AlbertTokenizer.from_pretrained(model_ckpt)
    model = AlbertForTokenClassification.from_pretrained(model_ckpt)

    # Load data sample
    sample_text = "HuggingFace is a company based in Paris and New York"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    model(**input_tokens)

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]
    varaint_name = model_ckpt.replace("-", "_")
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=f"pt_{varaint_name}_token_cls")
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from transformers import (
    Phi3Config,
    Phi3ForCausalLM,
    AutoTokenizer,
    Phi3ForTokenClassification,
    Phi3ForSequenceClassification,
)
import pytest
import forge


variants = ["microsoft/phi-3-mini-4k-instruct"]


@pytest.mark.parametrize("variant", variants)
def test_phi3_causal_lm(variant, test_device):

    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = Phi3ForCausalLM.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()

    # input_prompt
    input_prompt = "Africa is an emerging economy because"

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    inputs = [input_ids, attn_mask]

    compiled_model = forge.compile(model, sample_inputs=inputs)


@pytest.mark.parametrize("variant", variants)
def test_phi3_token_classification(variant, test_device):

    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)

    model = Phi3ForTokenClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]

    compiled_model = forge.compile(model, sample_inputs=inputs)


@pytest.mark.parametrize("variant", variants)
def test_phi3_sequence_classification(variant, test_device):

    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config_dict["pad_token_id"] = None
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    model = Phi3ForSequenceClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()

    # input_prompt
    input_prompt = "the movie was great!"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")
    inputs = [inputs["input_ids"]]

    compiled_model = forge.compile(model, sample_inputs=inputs)
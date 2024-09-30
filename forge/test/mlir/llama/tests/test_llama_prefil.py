# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

import forge
from test.mlir.llama.utils.utils import load_model

def decode_on_cpu(model, tokenizer, input_ids, hidden_states, max_new_tokens):
    for i in range(max_new_tokens):
        # Use only the hidden state of the last token in the sequence
        print("Shape of the last hidden state: ", hidden_states.shape)
        next_token_logits = model.lm_head(hidden_states[:, -1, :])

        # Get the next token ID
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        # Stop if the EOS token is generated
        if next_token_id == tokenizer.eos_token_id:
            break

        # Update input_ids with the new token
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

        # Run the model again to get the new hidden state
        with torch.no_grad():
            transformer_outputs = model.model(
                input_ids=input_ids,    # Pass the entire updated sequence
                past_key_values=None,   # No caching
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden_states = transformer_outputs.last_hidden_state
    
    return input_ids, hidden_states


def test_llama_prefil_on_device_decode_on_cpu():
    """
    This function tests the inference of the Llama 3B model splitted into two parts:
    - The first part is the prefilling of the model on the device.
    - The second part is the decoding of the model on the CPU.
    """
    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print("length of input_ids before text generation: ", len(input_ids[0]))
    # This is the part of the model needed for prefil; model without last Linear layer (lm_head)
    model_decoder = model.get_decoder()
    model_decoder.config.use_cache = False
    model_decoder.config.output_attentions = False
    model_decoder.config.output_hidden_states = False
    model_decoder.config.return_dict = True
    compiled_decoder = forge.compile(model_decoder, sample_inputs=input_ids)

    # Prefill Phase - Process the initial prompt
    # Using torch.no_grad() to prevent gradient calculation
    with torch.no_grad():
        transformer_outputs = compiled_decoder(input_ids)
        # Get hidden states for all tokens from the last "transformer layer"
        hidden_states = transformer_outputs.last_hidden_state

    # Decode Phase - Generate new tokens
    max_new_tokens = 46
    hidden_states = hidden_states.to("cpu")
    input_ids, hidden_states = decode_on_cpu(model, tokenizer, input_ids, hidden_states, max_new_tokens)

    # Generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(generated_text)
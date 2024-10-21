# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

import forge
from forge.op.eval.common import compare_with_golden_pcc
from test.mlir.llama.utils.utils import load_model_and_tokenizer


# @pytest.mark.xfail()
def test_llama_inference():
    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model, _ = load_model_and_tokenizer(model_path=model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Sanity run
    generation_output = framework_model.generate(input_ids=input_ids, max_new_tokens=32)
    print(tokenizer.decode(generation_output[0]))

    # Compile the model
    compiled_model = forge.compile(framework_model, input_ids)


@pytest.mark.skip(reason="No need to run in CI, this is PoC that should be mapped to work on device.")
def test_llama_inference_no_cache_cpu():
    """
    This function tests the inference of the Llama 3B model without using a past-cache (KV cache).
    It generates text token by token, which can slow down over time as the model has to compute
    all key-value (KV) pairs for each new token. The function demonstrates how to load the model
    and tokenizer, prepare an input prompt, and generate a sequence of tokens until a specified
    maximum number of new tokens is reached or an end-of-sequence token is encountered.
    """
    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model = load_model(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    max_new_tokens = 46
    generated_tokens = input_ids
    for i in range(max_new_tokens):
        logits = framework_model(input_ids)
        next_token_logits = logits[0][:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

    # Generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(generated_text)


@pytest.mark.skip(reason="No need to run in CI, this is PoC that should be mapped to work on device.")
def test_llama_inference_cache_cpu():
    """
    This function tests the inference of the Llama 3B model using a past-cache (KV cache).
    By utilizing cached key-value (KV) pairs, the model can generate text more efficiently
    as it doesn't need to recompute KV pairs for previously generated tokens. The function
    demonstrates how to load the model and tokenizer, prepare an input prompt, and generate
    a sequence of tokens until a specified maximum number of new tokens is reached or an
    end-of-sequence token is encountered.

    Steps:
    1. Load the Llama 3B model and tokenizer with caching enabled.
    2. Prepare an input prompt and convert it to input IDs.
    3. Initialize past key-values and other necessary inputs.
    4. Perform a prefill step to get the initial logits and past key-values.
    5. Generate tokens iteratively, updating the past key-values and input IDs.
    6. Decode the generated tokens into text and print the result.
    """
    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model = load_model(model_path, use_cache=True)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_length = input_ids.size(-1)

    # Prepare other inputs
    past_key_values = None
    pkv_length = len(past_key_values[0][0].shape[2]) if past_key_values else 0
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(pkv_length, seq_length + pkv_length, dtype=torch.long).unsqueeze(0)

    inputs = (input_ids, attention_mask, position_ids, past_key_values)

    # Prefill
    logits, past_key_values = framework_model(*inputs)

    # Fetch token for 0 iteration
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    max_new_tokens = 46
    generated_tokens = input_ids
    for i in range(max_new_tokens):
        logits, past_key_values = framework_model(input_ids=next_token.unsqueeze(0), past_key_values=past_key_values)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break
        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    # Generated text
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(generated_text)


@pytest.mark.xfail()
def test_llama_prefill_on_cpu_decode_on_tt_no_cache():

    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model, tokenizer = load_model_and_tokenizer(model_path=model_path, use_cache=False)
    tokenizer.pad_token_id = framework_model.config.pad_token_id

    # Prepare input sentence
    max_sequence_length = 53
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_sequence_length,
        pad_to_max_length=True,
        truncation=True,
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    non_padding_seq_len = int(torch.sum(attention_mask))
    padding_seq_len = attention_mask.shape[1] - non_padding_seq_len

    # Run Prefill on CPU with no cache to get the initial logits
    logits = framework_model(input_ids=input_ids, attention_mask=attention_mask)

    # Take the last non padding token index in logits for the next token
    next_token_logits = logits[0][:, non_padding_seq_len - 1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    # Update the input_ids with predicted token from prefill in the first padding token index
    input_ids[:, non_padding_seq_len] = next_token
    attention_mask[:, non_padding_seq_len] = 1

    non_padding_seq_len = int(torch.sum(attention_mask))
    padding_seq_len = attention_mask.shape[1] - non_padding_seq_len

    # Compile the model on TT
    compiled_model = forge.compile(framework_model, sample_inputs=[input_ids, attention_mask])

    # Run decode stage on TT device and generate tokens by appending predicted token into sequence of input tokens
    # untill the a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
    max_new_tokens = padding_seq_len
    for idx in range(max_new_tokens):

        model_inputs = [input_ids, attention_mask]

        # CPU Inference
        framework_output = framework_model(input_ids=input_ids, attention_mask=attention_mask)

        # Run on TT device
        tt_output = compiled_model(*model_inputs)
        tt_output = [tt_out.to("cpu") for tt_out in tt_output]

        # Validate TT result with Framework
        assert all(
            [
                compare_with_golden_pcc(golden=fw_out, calculated=tt_out, pcc=0.99)
                for fw_out, tt_out in zip(framework_output, tt_output)
            ]
        )

        next_token_index = non_padding_seq_len + idx
        next_token_logits = tt_output[0][:, next_token_index - 1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

        input_ids[:, next_token_index] = next_token
        attention_mask[:, next_token_index] = 1

    # Generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("generated_text=", generated_text)


@pytest.mark.xfail()
def test_llama_prefill_on_cpu_decode_on_tt_cache():

    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model, tokenizer = load_model_and_tokenizer(model_path=model_path, use_cache=True, return_dict=True)

    class LlamaModelWrapper(torch.nn.Module):
        """
        In LlamaModelWrapper class, forward function takes single input token (i.e last predicted token)
        and past key values from the previous iteration and return logits and past key values
        Args:
            input_id (`torch.Tensor`) - shape of (batch_size, 1)
            past_key_values (`List[List[torch.Tensor, torch.Tensor]]`) - key/values shape - (batch_size, num_of_key_values_heads, key_value_seq_len, head_dim)
        Returns:
            outputs - Logits of shape (batch_size, 1, vocab_size) and past_key_values (`List[List[torch.Tensor, torch.Tensor]]`)
            past key/values tensor shape - (batch_size, num_of_key_values_heads, key_value_seq_len + 1, head_dim)
        """

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_id, past_key_values):
            model_outputs = self.model(input_ids=input_id, past_key_values=past_key_values)
            return model_outputs.logits, model_outputs.past_key_values

    llama_model = LlamaModelWrapper(framework_model)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Run Prefill on CPU with cache to get the initial logits and past key-values
    prefill_output = framework_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    next_token_logits = prefill_output.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    generated_tokens = inputs.input_ids
    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    model_inputs = [next_token.unsqueeze(0), prefill_output.past_key_values]

    # Run decode stage on TT device and generate tokens by passing the last predicted token and the past key values.
    # untill the a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
    max_new_tokens = 32
    for _ in range(max_new_tokens):

        # CPU Inference
        model_outputs = llama_model(model_inputs[0], model_inputs[1])

        # TT will return the logits and past key values as list of tensor, so flattening
        # framework output past key values from List(List(Key1, Values1), ... , List(Key26, Values26)) to
        # List(Key1, Values1, ... , Key26, Values26) for comparing the Framework and TT output in similar fashion.
        framework_output = [model_outputs[0]]
        for k, v in model_outputs[1]:
            framework_output.append(k)
            framework_output.append(v)

        # Compile the model
        compiled_model = forge.compile(llama_model, sample_inputs=model_inputs)

        # Run on TT device
        tt_output = compiled_model(*model_inputs)
        tt_output = [tt_out.to("cpu") for tt_out in tt_output]

        # Validate TT result with Framework
        assert all(
            [
                compare_with_golden_pcc(golden=fw_out, calculated=tt_out, pcc=0.99)
                for fw_out, tt_out in zip(framework_output, tt_output)
            ]
        )
        logits = tt_output[0]
        past_key_values = tt_output[1:]

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

        model_inputs = [next_token.unsqueeze(0)]

        # Formate the past key values from List(Key1, Values1, ... , Key26, Values26) to
        # List(List(Key1, Values1), ... , List(Key26, Values26))
        model_inputs.append([past_key_values[idx : idx + 2] for idx in range(0, len(past_key_values), 2)])
        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    # Generated text
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("generated_text=", generated_text)


# The test is just workaround for checking with/without padding.
# @pytest.mark.xfail()
def test_llama_prefill_on_cpu_decode_on_tt_cache_padding(apply_padding_on_past_key_values=False):

    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model, tokenizer = load_model_and_tokenizer(model_path=model_path, use_cache=True, return_dict=True)

    class LlamaModelWrapper(torch.nn.Module):
        """
        In LlamaModelWrapper class, forward function takes single input token (i.e last predicted token)
        and past key values from the previous iteration and return logits and past key values
        Args:
            input_id (`torch.Tensor`) - shape of (batch_size, 1)
            past_key_values (`List[List[torch.Tensor, torch.Tensor]]`) - key/values shape - (batch_size, num_of_key_values_heads, key_value_seq_len, head_dim)
        Returns:
            outputs - Logits of shape (batch_size, 1, vocab_size) and past_key_values (`List[List[torch.Tensor, torch.Tensor]]`)
            past key/values tensor shape - (batch_size, num_of_key_values_heads, key_value_seq_len + 1, head_dim)
        """

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_id, past_key_values, attention_mask):
            model_outputs = self.model(
                input_ids=input_id, past_key_values=past_key_values, attention_mask=attention_mask
            )
            return model_outputs.logits, model_outputs.past_key_values, model_outputs.hidden_states

    llama_model = LlamaModelWrapper(framework_model)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Run Prefill on CPU with cache to get the initial logits and past key-values
    prefill_output = framework_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    next_token_logits = prefill_output.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    generated_tokens = inputs.input_ids
    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    past_key_values_list = [[k, v] for k, v in prefill_output.past_key_values]

    model_inputs = [next_token.unsqueeze(0), past_key_values_list]
    non_padding_seq_len = past_key_values_list[0][0].shape[-2]

    # Padding on past key values can be enabled by setting apply_padding_on_past_key_values to True.
    max_new_tokens = 1
    if apply_padding_on_past_key_values:
        padding_seq_len = max_new_tokens

        # Zero Pad past key values in key_value_seq_len(i.e -2) dimension
        # Before padding past key values tensor shape -> (batch_size, num_of_key_values_heads, key_value_seq_len, head_dim)
        # After Padding Past key value tensor shape -> (batch_size, num_of_key_values_heads, key_value_seq_len + padding_seq_len, head_dim)
        for idx, (k, v) in enumerate(model_inputs[1]):
            model_inputs[1][idx][0] = torch.cat(
                [
                    k,
                    torch.zeros(k.shape[-4], k.shape[-3], max_new_tokens, k.shape[-1]).to(k.dtype),
                ],
                dim=-2,
            )
            model_inputs[1][idx][1] = torch.cat(
                [
                    v,
                    torch.zeros(v.shape[-4], v.shape[-3], max_new_tokens, v.shape[-1]).to(k.dtype),
                ],
                dim=-2,
            )
    kv_states_after_prefil = model_inputs[1]
    # # Compile the model
    # compiled_model = forge.compile(llama_model, sample_inputs=model_inputs)

    # Run decode stage on TT device and generate tokens by passing the last predicted token and the past key values.
    # untill the a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
    attention_mask = inputs.attention_mask
    attention_mask = torch.cat(
        [
            attention_mask,  # Existing mask
            torch.zeros((attention_mask.shape[0], 1), dtype=torch.long),  # Mask for new token
        ],
        dim=1,
    )
    # Loop to generate new tokens
    last_k_state = []
    last_v_state = []
    for max_new_tokens_idx in range(max_new_tokens):
        if apply_padding_on_past_key_values:
            padded_att_mask = torch.cat(
                [
                    attention_mask,  # Existing mask
                    torch.full(
                        (attention_mask.shape[0], max_new_tokens - max_new_tokens_idx),
                        float("-inf"),  # Use -inf to mask new tokens
                        dtype=torch.float32,
                    ),
                ],
                dim=1,
            )
            padded_att_mask[:, -1] = 0
            model_outputs = llama_model(model_inputs[0], model_inputs[1], padded_att_mask)
        else:
            model_outputs = llama_model(model_inputs[0], model_inputs[1], attention_mask)

        # CPU Inference
        # model_outputs = llama_model(model_inputs[0], model_inputs[1])

        # TT will return the logits and past key values as list of tensor, so flattening
        # framework output past key values from List(List(Key1, Values1), ... , List(Key26, Values26)) to
        # List(Key1, Values1, ... , Key26, Values26) for comparing the Framework and TT output in similar fashion.
        framework_output = [model_outputs[0]]
        ind = 0
        for k, v in model_outputs[1]:
            if ind == 0:
                print(" K vec for ind: ", max_new_tokens_idx, "  ", k[0, 0, -1, :7])
            assert torch.allclose(
                k[:, :, :-1, :], model_inputs[1][ind][0]
            ), f"Tensor at index {ind} in past key does not match"
            assert torch.allclose(
                v[:, :, :-1, :], model_inputs[1][ind][1]
            ), f"Tensor at index {ind} in past value does not match"
            ind += 1
            framework_output.append(k)
            framework_output.append(v)

        # # Run on TT device
        # tt_output = compiled_model(*model_inputs)
        # tt_output = [tt_out.to("cpu") for tt_out in tt_output]

        # # Validate TT result with Framework
        # assert all([compare_with_golden_pcc(golden=fw_out, calculated=tt_out, pcc=0.99) for fw_out, tt_out in zip(framework_output, tt_output)])

        logits = framework_output[0]
        past_key_values = framework_output[1:]

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        # print("next_token=", next_token)

        if next_token == tokenizer.eos_token_id:
            break

        model_inputs = [next_token.unsqueeze(0)]

        # Formate the past key values from List(Key1, Values1, ... , Key26, Values26) to
        # List(List(Key1, Values1), ... , List(Key26, Values26))
        model_inputs.append([past_key_values[idx : idx + 2] for idx in range(0, len(past_key_values), 2)])
        # **Update the attention mask** to account for the new token
        attention_mask = torch.cat(
            [
                attention_mask,  # Existing mask
                torch.zeros((attention_mask.shape[0], 1), dtype=torch.long),  # Mask for new token
            ],
            dim=1,
        )
        for idx in range(len(model_inputs[1])):
            last_k_state.append(torch.clone(model_inputs[1][idx][0][:, :, -1, :]))
            last_v_state.append(torch.clone(model_inputs[1][idx][1][:, :, -1, :]))
        if apply_padding_on_past_key_values:
            for idx in range(len(model_inputs[1])):
                model_inputs[1][idx][0][:, :, non_padding_seq_len + max_new_tokens_idx, :] = model_inputs[1][idx][0][
                    :, :, -1, :
                ]
                model_inputs[1][idx][0] = model_inputs[1][idx][0][:, :, :-1, :]
                model_inputs[1][idx][1][:, :, non_padding_seq_len + max_new_tokens_idx, :] = model_inputs[1][idx][1][
                    :, :, -1, :
                ]
                model_inputs[1][idx][1] = model_inputs[1][idx][1][:, :, :-1, :]
                # check that all elements of subtensor model_inputs[1][idx][0][:, :, non_padding_seq_len + max_new_tokens_idx + 1:, :] are zero
                assert torch.all(model_inputs[1][idx][0][:, :, non_padding_seq_len + max_new_tokens_idx + 1 :, :] == 0)
                assert torch.all(model_inputs[1][idx][1][:, :, non_padding_seq_len + max_new_tokens_idx + 1 :, :] == 0)
                # model_inputs[1][idx][0][:, :, non_padding_seq_len + max_new_tokens_idx + 1:, :] = 0
                # model_inputs[1][idx][1][:, :, non_padding_seq_len + max_new_tokens_idx + 1:, :] = 0
        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)
    print("generated_tokens[0]=", generated_tokens[0])
    # Generated text
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("generated_text=", generated_text)
    return last_k_state, last_v_state, kv_states_after_prefil, non_padding_seq_len, model_outputs[2]


def compare_tensors(tensor1, tensor2, name="Tensor"):
    # Calculate the absolute difference
    abs_diff = torch.abs(tensor1 - tensor2)

    # Compute key statistics on the difference
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    total_elements = tensor1.numel()  # Total number of elements in the tensor
    scaled_diff = abs_diff.sum().item() / total_elements  # Average difference

    # Print results
    print(f"{name} - Max Difference: {max_diff}")
    print(f"{name} - Mean Difference: {mean_diff}")
    print(f"{name} - Scaled Difference (per element): {scaled_diff}")

    # Return True if tensors are sufficiently close (based on a threshold)
    close = torch.allclose(tensor1, tensor2, rtol=1e-3, atol=1e-5)
    print(f"{name} - Are tensors close: {close}")

    return close


def test_compare_states():
    (
        last_k_state_pad,
        last_v_state_pad,
        kv_states_after_prefil_pad,
        non_padding_begin_seq_len,
        hidden_states_pad,
    ) = test_llama_prefill_on_cpu_decode_on_tt_cache_padding(apply_padding_on_past_key_values=True)
    (
        last_k_state_no_pad,
        last_v_state_no_pad,
        kv_states_after_prefil_no_pad,
        _,
        hidden_states_no_pad,
    ) = test_llama_prefill_on_cpu_decode_on_tt_cache_padding(apply_padding_on_past_key_values=False)

    print("Comparing past key values after prefill with and without padding")
    for idx in range(len(kv_states_after_prefil_pad)):
        assert torch.allclose(
            kv_states_after_prefil_pad[idx][0][:, :, :non_padding_begin_seq_len, :],
            kv_states_after_prefil_no_pad[idx][0],
        )
        assert torch.allclose(
            kv_states_after_prefil_pad[idx][1][:, :, :non_padding_begin_seq_len, :],
            kv_states_after_prefil_no_pad[idx][1],
        )

    for decoder_idx in range(len(last_k_state_pad)):
        print(f"Comparing Key, Value and Hidden state vectors for last generated token for decoder {decoder_idx}\n")
        compare_tensors(last_k_state_pad[decoder_idx], last_k_state_no_pad[decoder_idx], name="Key Tensor")
        compare_tensors(last_v_state_pad[decoder_idx], last_v_state_no_pad[decoder_idx], name="Value Tensor")
        # compare_tensors(hidden_states_pad[decoder_idx], hidden_states_no_pad[decoder_idx], name="Hidden State Tensor")

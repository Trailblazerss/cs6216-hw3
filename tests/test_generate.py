import os
import platform
import pytest
import warnings

import torch

from transformers import AutoTokenizer
from model import PythiaForCausalLM

from generate import prefill, decode
from model import KVCache


@pytest.fixture
def tokerizer_and_model():
    precision = torch.bfloat16
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "../checkpoints/EleutherAI/pythia-410m")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, clean_up_tokenization_spaces=True)
    model = PythiaForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=precision)
    return tokenizer, model


def test_prefill(tokerizer_and_model):
    tokenizer, model = tokerizer_and_model
    input_text = "My name is "
    encoded = tokenizer(input_text, return_tensors="pt")
    generator = torch.Generator().manual_seed(6216)
    next_token, kvcaches = prefill(model, encoded["input_ids"], encoded['attention_mask'], generator=generator)

    assert len(next_token.size()) == 0
    assert len(kvcaches) == model.config.num_hidden_layers and all([isinstance(kvcache, KVCache) for kvcache in kvcaches])

    if platform.system() == 'Darwin':
        assert next_token.item() == 2015
    elif platform.system() in ['Linux', 'Windows']:
        assert next_token.item() == 187
    else:
        warnings.warn(f"Unknown platform: {platform.system()}")


def test_decode(tokerizer_and_model):
    tokenizer, model = tokerizer_and_model
    input_text = "My name is "
    encoded = tokenizer(input_text, return_tensors="pt")
    generator = torch.Generator().manual_seed(6216)
    
    # Prefill step, as before
    next_token, kvcaches = prefill(model, encoded["input_ids"], encoded['attention_mask'], generator=generator)

    assert len(next_token.size()) == 0
    assert len(kvcaches) == model.config.num_hidden_layers and all(isinstance(kvcache, KVCache) for kvcache in kvcaches)
    
    # Platform-dependent check
    if platform.system() == 'Darwin':
        assert next_token.item() == 2015
    elif platform.system() in ['Linux', 'Windows']:
        assert next_token.item() == 187
    else:
        warnings.warn(f"Unknown platform: {platform.system()}")

    # Call decode and verify the stopping condition with EOS token handling
    generated_tokens = decode(model, kvcaches, next_token.view(1, -1), encoded['attention_mask'], num_new_tokens=5, generator=generator, tokenizer=tokenizer)
    assert len(generated_tokens) <= 5  # Should stop generating at EOS token or end of loop
    # Additional checks on content of generated tokens if needed

# def test_decode(tokerizer_and_model):
#     tokenizer, model = tokerizer_and_model
#     input_text = "My name is "
#     encoded = tokenizer(input_text, return_tensors="pt")
#     generator = torch.Generator().manual_seed(6216)

#     # Prefill step, as before
#     next_token, kvcaches = prefill(model, encoded["input_ids"], encoded['attention_mask'], generator=generator)

#     assert len(next_token.size()) == 0
#     assert len(kvcaches) == model.config.num_hidden_layers and all(isinstance(kvcache, KVCache) for kvcache in kvcaches)

#     # Platform-dependent check, allowing flexibility with possible values
#     if platform.system() == 'Darwin':
#         assert next_token.item() == 2015
#     elif platform.system() in ['Linux', 'Windows']:
#         assert next_token.item() in [187, 6625]  # Allow for both 187 and 6625
#     else:
#         warnings.warn(f"Unknown platform: {platform.system()}")

#     # Decode step
#     generated_tokens = decode(
#         model, kvcaches, next_token.view(1, -1), encoded['attention_mask'], 
#         num_new_tokens=5, generator=generator, tokenizer=tokenizer
#     )

#     # Check for the presence of tokens in the generated sequence (adjust as per actual output)
#     assert len(generated_tokens) == 5
#     assert all(isinstance(token, int) for token in generated_tokens)

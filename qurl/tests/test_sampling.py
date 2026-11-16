"""qurl.sampling: batched sampling with the behavior policy's log-probabilities, and the training forward."""
import pytest
import torch

from qurl.fakequant import int8_rollout
from qurl.sampling import generate, sequence_logprobs, stop_token_ids
from qurl_test_support import tiny_qwen2, tiny_tokenizer

PROMPTS = ["Natalia sold clips to 48 of her friends", "How many clips"]   # different lengths: left padding


@pytest.fixture(scope="module")
def tok():
    t = tiny_tokenizer()
    t.padding_side = "left"
    return t


@pytest.fixture(scope="module")
def model():
    return tiny_qwen2(seed=0).eval()


def sample(model, tok, seed=0, **kw):
    kw.setdefault("max_new", 12)
    return generate(model, tok, PROMPTS, generator=torch.Generator().manual_seed(seed), **kw)


@pytest.mark.parametrize("quantized", [False, True])
def test_logprobs_match_a_full_forward(model, tok, quantized):
    """mu, recorded while sampling with the KV cache, equals the log-probabilities of a full forward pass."""
    ctx = int8_rollout(model) if quantized else torch.no_grad()
    with ctx, torch.no_grad():
        roll = sample(model, tok, n=3)
        full = sequence_logprobs(model, roll.input_ids, roll.attention_mask, roll.response)
    assert roll.response.shape[0] == 6 and roll.input_ids.shape[1] == roll.prompt_len + roll.response.shape[1]
    torch.testing.assert_close(roll.logprobs, full * roll.mask, rtol=0, atol=1e-5)


def test_temperature_scales_the_policy(model, tok):
    with torch.no_grad():
        roll = sample(model, tok, n=2, temperature=0.7)
        full = sequence_logprobs(model, roll.input_ids, roll.attention_mask, roll.response, temperature=0.7)
    torch.testing.assert_close(roll.logprobs, full * roll.mask, rtol=0, atol=1e-5)


def test_seeded_sampling_is_reproducible(model, tok):
    a, b, c = sample(model, tok, seed=1, n=2), sample(model, tok, seed=1, n=2), sample(model, tok, seed=2, n=2)
    assert torch.equal(a.response, b.response) and torch.equal(a.logprobs, b.logprobs)
    assert not torch.equal(a.response, c.response)


def test_greedy_takes_the_argmax(model, tok):
    roll = sample(model, tok, temperature=0.0, max_new=8)
    with torch.no_grad():
        pos = (roll.attention_mask.cumsum(-1) - 1).clamp(min=0)
        logits = model(input_ids=roll.input_ids, attention_mask=roll.attention_mask, position_ids=pos).logits
    best = logits[:, roll.prompt_len - 1:-1].argmax(-1)
    assert torch.equal(best[roll.mask], roll.response[roll.mask])


def test_stop_tokens_end_responses(model, tok):
    stop = list(range(48))   # half of the vocabulary, so that every response ends within a few tokens
    roll = sample(model, tok, n=3, max_new=30, stop_ids=stop)
    for resp, mask, lp in zip(roll.response, roll.mask, roll.logprobs):
        n = int(mask.sum())
        assert mask[:n].all() and not mask[n:].any()          # a prefix: the response up to its end token
        assert int(resp[n - 1]) in stop and not any(int(t) in stop for t in resp[:n - 1])
        assert (resp[n:] == tok.pad_token_id).all() and (lp[n:] == 0).all()
    assert roll.response.shape[1] < 30   # sampling stops once every response has ended


def test_left_padding_does_not_change_a_greedy_response(model, tok):
    alone = generate(model, tok, PROMPTS[1:], max_new=8, temperature=0.0)
    batched = generate(model, tok, PROMPTS, max_new=8, temperature=0.0)
    n = int(alone.mask[0].sum())
    assert torch.equal(alone.response[0, :n], batched.response[1, :n])


def test_needs_left_padding_and_finds_stop_tokens(model):
    t = tiny_tokenizer()
    assert stop_token_ids(t) == [1, 2]   # <pad> and <eos>
    t.padding_side = "right"
    with pytest.raises(ValueError, match="pads on the left"):
        generate(model, t, PROMPTS, max_new=2)

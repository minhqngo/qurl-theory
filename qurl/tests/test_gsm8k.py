"""qurl.gsm8k: problems, prompt and the exact-match reward (offline: local files only)."""
import json

import pytest

from qurl import gsm8k


@pytest.mark.parametrize("text, want", [
    ("so the total is \\boxed{72}.", 72.0),
    ("first \\boxed{3}, finally \\boxed{1,234}", 1234.0),
    ("She earns $5.50 per hour.", 5.5),
    ("The answer is 18.", 18.0),
    ("\\boxed{\\$18}", 18.0),
    ("\\boxed{-4}", -4.0),
    ("no number here", None),
    ("\\boxed{x + 1}", None),
])
def test_extract_answer(text, want):
    assert gsm8k.extract_answer(text) == want


def test_reward():
    assert gsm8k.reward("Natalia sold 48 + 24 = \\boxed{72} clips.", 72.0) == 1.0
    assert gsm8k.reward("Natalia sold \\boxed{70} clips.", 72.0) == 0.0
    assert gsm8k.reward("I do not know.", 72.0) == 0.0


def test_parse_and_load_local_split(tmp_path):
    rows = [{"question": "How many?", "answer": "48 / 2 = 24\n#### 1,024"},
            {"question": "Total?", "answer": "#### -3"}]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    problems = gsm8k.load_split("train", str(tmp_path))   # present: no download
    assert problems == [gsm8k.Problem("How many?", 1024.0), gsm8k.Problem("Total?", -3.0)]


def test_prompt_asks_for_a_boxed_answer():
    assert gsm8k.prompt("How many?") == "How many?\nPlease reason step by step, and put your final answer within " \
                                        "\\boxed{}."

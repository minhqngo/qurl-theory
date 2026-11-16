"""GSM8K for QuRL's training loop: the problems, the prompt and the exact-match reward.

The splits are the official JSON-lines files of github.com/openai/grade-school-math (7,473 training and 1,319 test
problems), downloaded on first use into a cache directory. A problem's answer is the number after "####". The prompt
asks for the final answer in \\boxed{}; the reward is 1 if the last boxed value (or, without one, the last number in
the response) equals the answer, else 0.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass

URLS = {split: ("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"
                f"{split}.jsonl") for split in ("train", "test")}
DEFAULT_DIR = "~/.cache/qurl/gsm8k"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


@dataclass(frozen=True)
class Problem:
    question: str
    answer: object        # GSM8K: a float


def parse_jsonl(path: str) -> list[Problem]:
    """GSM8K's JSON lines: {"question": ..., "answer": "... #### 42"}."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out.append(Problem(row["question"], float(row["answer"].split("####")[-1].replace(",", "").strip())))
    return out


def load_split(split: str, data_dir: str | None = None) -> list[Problem]:
    """The "train" or "test" split, from <data_dir>/<split>.jsonl; downloaded there if missing."""
    path = os.path.join(os.path.expanduser(data_dir or DEFAULT_DIR), f"{split}.jsonl")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(URLS[split], timeout=60) as r:
            data = r.read()
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    return parse_jsonl(path)


def prompt(question: str) -> str:
    """The user message for a problem (the training loop wraps it in the model's chat template)."""
    return question + SUFFIX


def extract_answer(text: str) -> float | None:
    """The last \\boxed{...} value of a response, else its last number; None if there is none."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        s = boxed[-1]
    else:
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
        s = numbers[-1] if numbers else ""
    s = s.replace(",", "").replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def reward(text: str, answer: object) -> float:
    """1.0 if the response's final answer equals `answer`, else 0.0."""
    pred = extract_answer(text)
    return float(pred is not None and abs(pred - float(answer)) < 1e-6)

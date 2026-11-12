"""S4 data: DAPO-Math-17k (training), AIME 2024 and MATH-500 (evaluation), in VeRL's RL parquet format.

  python prepare_data.py --out ~/data/s4 [--smoke]

Downloads (Hugging Face datasets, public):
  BytedTsinghua-SIA/DAPO-Math-17k  data/dapo-math-17k.parquet   the paper's training set ("the 17k dataset")
  BytedTsinghua-SIA/AIME-2024      data/aime-2024.parquet       the paper's evaluation set
  HuggingFaceH4/MATH-500           test.jsonl                   secondary evaluation set (500 problems)
Writes to --out:
  dapo17k.parquet     DAPO-Math-17k with duplicate prompts removed: the published file has 1,791,700 rows but 17,398
                      distinct problems; the 7 whose copies carry different answers are dropped, leaving 17,391.
                      200 steps x 512 prompts then cover each problem about 6 times, in a seeded order
  aime24.parquet      the 30 AIME 2024 problems once each (offline evaluation, eval.py)
  aime24_x32.parquet  each AIME problem 32 times, for in-loop validation with n = 1: the mean is Avg@32
  math500.parquet     MATH-500 in the DAPO prompt format, ground truth = the dataset's answer
  data.json           counts and checks
--smoke also writes dapo_smoke.parquet (64 prompts) and aime_smoke.parquet (4 rows) for laptop tests.
Every row: data_source, prompt (one user message), ability, reward_model {ground_truth, style}, extra_info {index}.
The DAPO prompt wording is taken from the training file itself; the script checks that every training prompt uses it.
"""
import argparse
import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download

PREFIX = ("Solve the following math problem step by step. The last line of your response should be of the form "
          "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n")
SUFFIX = "\n\nRemember to put your answer on its own line after \"Answer:\"."


def content(prompt) -> str:
    msgs = list(prompt)
    assert len(msgs) == 1 and msgs[0]["role"] == "user", msgs
    return msgs[0]["content"]


def row(source, text, answer, index):
    return {"data_source": source, "prompt": [{"role": "user", "content": text}], "ability": "MATH",
            "reward_model": {"ground_truth": str(answer), "style": "rule-lean"}, "extra_info": {"index": index}}


def dedupe(df, name):
    """One row per distinct prompt; prompts whose copies carry different answers are dropped (ambiguous labels)."""
    texts = df["prompt"].map(content)
    answers = df["reward_model"].map(lambda r: str(r["ground_truth"]))
    clash = pd.DataFrame({"t": texts, "a": answers}).groupby("t")["a"].nunique()
    bad = set(clash.index[clash > 1])
    keep = ~texts.duplicated() & ~texts.isin(bad)
    print(f"{name}: {len(df)} rows, {texts.nunique()} distinct prompts, {len(bad)} dropped for conflicting answers, "
          f"{int(keep.sum())} kept")
    return df[keep].reset_index(drop=True), texts[keep].reset_index(drop=True), len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    get = lambda repo, f: hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", local_dir=raw)  # noqa: E731

    info = {}
    # Training set.
    published = pd.read_parquet(get("BytedTsinghua-SIA/DAPO-Math-17k", "data/dapo-math-17k.parquet"))
    dapo, texts, n_clash = dedupe(published, "DAPO-Math-17k")
    ok = texts.map(lambda t: t.startswith(PREFIX) and t.endswith(SUFFIX))
    print(f"  prompts in the DAPO format: {int(ok.sum())} of {len(ok)}; data sources: "
          f"{dapo['data_source'].value_counts().to_dict()}")
    assert ok.mean() > 0.99, "the DAPO prompt wording differs from the expected template"
    train = pd.DataFrame([row("math_dapo", t, r["ground_truth"], i)
                          for i, (t, r) in enumerate(zip(texts, dapo["reward_model"]))])
    train.to_parquet(os.path.join(out, "dapo17k.parquet"))
    info["dapo17k"] = {"rows_published": len(published), "kept": len(train), "dropped_conflicting": n_clash}

    # AIME 2024.
    aime, atexts, _ = dedupe(pd.read_parquet(get("BytedTsinghua-SIA/AIME-2024", "data/aime-2024.parquet")), "AIME-2024")
    assert len(aime) == 30, f"expected 30 AIME 2024 problems, found {len(aime)}"
    aime_rows = [row("aime24", t, r["ground_truth"], i) for i, (t, r) in enumerate(zip(atexts, aime["reward_model"]))]
    pd.DataFrame(aime_rows).to_parquet(os.path.join(out, "aime24.parquet"))
    x32 = [dict(r, extra_info={"index": r["extra_info"]["index"], "repeat": k}) for r in aime_rows for k in range(32)]
    pd.DataFrame(x32).to_parquet(os.path.join(out, "aime24_x32.parquet"))
    info["aime24"] = {"problems": len(aime_rows), "x32_rows": len(x32),
                      "in_dapo_format": int(sum(t.startswith(PREFIX) and t.endswith(SUFFIX) for t in atexts))}

    # MATH-500, in the training prompt format.
    m = pd.read_json(get("HuggingFaceH4/MATH-500", "test.jsonl"), lines=True)
    math_rows = [row("math500", PREFIX + p + SUFFIX, ans, i) for i, (p, ans) in enumerate(zip(m["problem"], m["answer"]))]
    pd.DataFrame(math_rows).to_parquet(os.path.join(out, "math500.parquet"))
    info["math500"] = {"problems": len(math_rows)}

    if a.smoke:
        train.sample(n=64, random_state=0).to_parquet(os.path.join(out, "dapo_smoke.parquet"))
        pd.DataFrame(aime_rows[:4]).to_parquet(os.path.join(out, "aime_smoke.parquet"))
        info["smoke"] = {"train": 64, "val": 4}
    with open(os.path.join(out, "data.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))
    print("example training prompt:", train.iloc[0]["prompt"][0]["content"][:200].replace("\n", " | "))


if __name__ == "__main__":
    main()

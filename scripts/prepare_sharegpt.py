"""Turn the raw ShareGPT JSON into a flat CSV of (prompt, in_tokens, out_tokens).

Uses the FIRST human->gpt exchange of each conversation. Token counts are
approximate (characters / 4), which is fine for a scheduling study because only
relative sizes matter.
"""
import json
import sys

import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/ShareGPT_V3_unfiltered_cleaned_split.json"
OUT = "data/sharegpt_pairs.csv"
MAX_PROMPT_CHARS = 3000          # keep the CSV small; features only need the start


def toks(text: str) -> int:
    return max(1, round(len(text) / 4))


with open(RAW, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for conv in data:
    turns = conv.get("conversations", [])
    for a, b in zip(turns, turns[1:]):
        if a.get("from") == "human" and b.get("from") == "gpt":
            prompt, reply = a.get("value", "").strip(), b.get("value", "").strip()
            if prompt and reply:
                rows.append((prompt[:MAX_PROMPT_CHARS], toks(prompt), toks(reply)))
            break                                   # first exchange only

df = pd.DataFrame(rows, columns=["prompt", "in_tokens", "out_tokens"])
df = df[(df.out_tokens >= 2) & (df.out_tokens <= 4096)].reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"conversations: {len(data)}  usable pairs: {len(df)}")
print(df[["in_tokens", "out_tokens"]].describe(percentiles=[.5, .9, .99]).round(1))
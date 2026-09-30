import os
import pandas as pd
from pathlib import Path

v4_dir = Path("data/csv/v4")
categories = [d for d in v4_dir.iterdir() if d.is_dir()]

# Helper: convert label to lowercase prefix (matches file naming convention)
def label_to_prefix(label):
    return label.lower().replace(" ", "_")

# Collect all Fewshot_pos from all categories
all_fewshot_pos = {}
for cat in categories:
    label = cat.name
    prefix = label_to_prefix(label)
    pos_file = cat / f"{prefix}_Fewshot_pos.csv"
    if pos_file.exists():
        df = pd.read_csv(pos_file, encoding="utf-8-sig")
        all_fewshot_pos[label] = df

# For each category, build neg4hotmap
for cat in categories:
    label = cat.name
    prefix = label_to_prefix(label)
    neg4hotmap = pd.DataFrame()

    # Collect all Fewshot_pos from OTHER categories (not this label)
    for other_label, df_pos in all_fewshot_pos.items():
        if other_label != label:
            neg4hotmap = pd.concat([neg4hotmap, df_pos], ignore_index=True)

    # Load this label's Fewshot_pos and Fewshot_neg to exclude
    own_pos_file = cat / f"{prefix}_Fewshot_pos.csv"
    own_neg_file = cat / f"{prefix}_Fewshot_neg.csv"

    if own_pos_file.exists():
        own_pos = pd.read_csv(own_pos_file, encoding="utf-8-sig")
        # Exclude by HackId + txHash
        mask = ~neg4hotmap.apply(lambda r: any(
            (neg4hotmap['HackId'] == r['HackId']) & (neg4hotmap['txHash'] == r['txHash'])
        ), axis=1)
        # Better: exclude rows that appear in own_pos
        exclude_ids = set(zip(own_pos['HackId'].astype(str), own_pos['txHash'].astype(str)))
        neg4hotmap = neg4hotmap[~neg4hotmap.apply(
            lambda r: (str(r['HackId']), str(r['txHash'])) in exclude_ids, axis=1
        )]

    if own_neg_file.exists():
        own_neg = pd.read_csv(own_neg_file, encoding="utf-8-sig")
        exclude_ids = set(zip(own_neg['HackId'].astype(str), own_neg['txHash'].astype(str)))
        neg4hotmap = neg4hotmap[~neg4hotmap.apply(
            lambda r: (str(r['HackId']), str(r['txHash'])) in exclude_ids, axis=1
        )]

    # Deduplicate
    neg4hotmap = neg4hotmap.drop_duplicates(subset=['HackId', 'txHash'])

    # Use lowercase prefix to match existing file naming convention
    prefix = label.lower().replace(" ", "_")
    out_file = cat / f"{prefix}_evaluation_neg4hotmap.csv"
    neg4hotmap.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[{label}] {len(neg4hotmap)} rows -> {out_file.name}")

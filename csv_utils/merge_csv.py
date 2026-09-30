# -*- coding: utf-8 -*-
"""
合并 data/csv/v4 下所有 csv:
  - 以 txHash(小写) 为主键去重合并（同一交易 = 同一项目）
  - 同一字段多个不同值用分隔符拼接去重
  - 额外保留 Categories / SourceFiles 溯源列
输出: data/csv/v4_merged.csv (utf-8-sig, Excel 友好)
"""
import csv, os
from collections import defaultdict, OrderedDict

ROOT = r"data\csv\v4"
OUT = r"data\csv\v4_merged.csv"

hash_to_rows = defaultdict(list)

for dirpath, _, filenames in os.walk(ROOT):
    for fn in filenames:
        if not fn.lower().endswith(".csv"):
            continue
        category = os.path.relpath(dirpath, ROOT)
        with open(os.path.join(dirpath, fn), "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                txh = (row.get("txHash") or "").strip()
                if not txh:
                    continue
                hash_to_rows[txh.lower()].append({
                    "category": category,
                    "file": os.path.join(category, fn).replace("\\", "/"),
                    "HackId": (row.get("HackId") or "").strip(),
                    "txHash": txh,
                    "Chain": (row.get("Chain") or "").strip(),
                    "Type": (row.get("Type") or "").strip(),
                    "Cause": (row.get("Cause") or "").strip(),
                    "URL": (row.get("Root cause URL") or "").strip(),
                })


def uniq_join(values, sep=" | "):
    """去空、保持首次出现顺序去重，用 sep 连接。"""
    seen = OrderedDict()
    for v in values:
        v = (v or "").strip()
        if v and v not in seen:
            seen[v] = None
    return sep.join(seen.keys())


# 合并
merged = []
for h, recs in hash_to_rows.items():
    merged.append({
        "HackId": uniq_join([r["HackId"] for r in recs], " | "),
        "txHash": recs[0]["txHash"],
        "Chain": uniq_join([r["Chain"] for r in recs]),
        "Type": uniq_join([r["Type"] for r in recs], " | "),
        "Cause": uniq_join([r["Cause"] for r in recs], " || "),
        "Root cause URL": uniq_join([r["URL"] for r in recs], " || "),
        "Categories": uniq_join([r["category"] for r in recs], ", "),
        "SourceFiles": uniq_join([r["file"] for r in recs], "; "),
    })


def sort_key(r):
    # 按主 HackId 数值升序，再按 txHash
    hid = r["HackId"].split(" | ")[0]
    try:
        n = int(hid)
    except ValueError:
        n = 1 << 30
    return (n, r["txHash"])

merged.sort(key=sort_key)

fields = ["HackId", "txHash", "Chain", "Type", "Cause",
          "Root cause URL", "Categories", "SourceFiles"]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(merged)

print(f"输入总行数: {sum(len(v) for v in hash_to_rows.values())}")
print(f"输出唯一交易数: {len(merged)}")
print(f"已写出: {OUT}")
print(f"\n前 3 行预览:")
for r in merged[:3]:
    print(f"  HackId={r['HackId']} | Chain={r['Chain']} | Type={r['Type']}")
    print(f"    txHash={r['txHash']}")
    print(f"    Categories={r['Categories']}")

"""
Dataset analysis: count unique categories, attributes, and relations.

Usage:
    python analyze_dataset.py
    python analyze_dataset.py --jsonl dataset/gpt_specs2_id_LLaVA_all.jsonl --out dataset_analysis.json
"""

import json
import argparse
from collections import Counter


def analyze(jsonl_path: str) -> dict:
    categories = Counter()
    attributes = Counter()
    relations  = Counter()
    n_clips    = 0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_clips += 1
            for c in rec.get("consts",      []):
                categories[c] += 1
            for u in rec.get("unary_kws",   []):
                attributes[u] += 1
            for b in rec.get("binary_kws",  []):
                relations[b]  += 1

    def to_sorted_list(counter: Counter) -> list:
        return [{"label": k, "count": v} for k, v in counter.most_common()]

    return {
        "n_clips":         n_clips,
        "n_categories":    len(categories),
        "n_attributes":    len(attributes),
        "n_relations":     len(relations),
        "categories":      to_sorted_list(categories),
        "attributes":      to_sorted_list(attributes),
        "relations":       to_sorted_list(relations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl", default="dataset/gpt_specs2_id_LLaVA_all.jsonl",
        help="Path to the JSONL dataset file",
    )
    parser.add_argument(
        "--out", default="dataset_analysis.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    print(f"Analysing {args.jsonl} ...")
    result = analyze(args.jsonl)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Done. Summary:")
    print(f"  clips      : {result['n_clips']:,}")
    print(f"  categories : {result['n_categories']:,} unique")
    print(f"  attributes : {result['n_attributes']:,} unique")
    print(f"  relations  : {result['n_relations']:,} unique")
    print(f"  Written to : {args.out}")


if __name__ == "__main__":
    main()

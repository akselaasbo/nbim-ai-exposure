"""Compare a classify.py output file with the hand labels in data/labels/sample.csv.

Prints accuracy (total, per size_group, per model confidence) with 95% Wilson intervals,
the share of knows_company=false, a confusion matrix, and every error with both views.
Saves the tables as CSV in results/, named after the output file.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from prompt import CATEGORIES, CONFIDENCE_LEVELS

SAMPLE_FILE = PROJECT_ROOT / "data" / "labels" / "sample.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion; behaves sensibly for small n."""
    if n == 0:
        return float("nan"), float("nan")
    p = correct / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def load_outputs(path: Path) -> pd.DataFrame:
    """One row per company: the last line with a valid output."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["output"] is None:
            continue
        rows.append({
            "company_id": record["company_id"],
            "model": record["model"],
            "prompt_version": record["prompt_version"],
            "attempts": record["attempts"],
            **{f"model_{k}": v for k, v in record["output"].items()},
        })
    return pd.DataFrame(rows).drop_duplicates("company_id", keep="last")


def accuracy_row(group: str, value: str, df: pd.DataFrame) -> dict:
    n, correct = len(df), int(df["correct"].sum())
    low, high = wilson_interval(correct, n)
    return {"group": group, "value": value, "n": n, "correct": correct,
            "accuracy": correct / n if n else float("nan"), "ci_low": low, "ci_high": high}


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [accuracy_row("total", "all", df)]
    for size_group, part in df.groupby("size_group"):
        rows.append(accuracy_row("size_group", size_group, part))
    for confidence in CONFIDENCE_LEVELS:
        rows.append(accuracy_row("model_confidence", confidence, df[df["model_confidence"] == confidence]))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classify.py output against hand labels.")
    parser.add_argument("results", help="JSONL file written by classify.py")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = parser.parse_args()

    path = Path(args.results)
    sample = pd.read_csv(SAMPLE_FILE, dtype=str, keep_default_na=False)
    if args.split != "all":
        sample = sample[sample["split"] == args.split]
    outputs = load_outputs(path)

    df = sample.merge(outputs, on="company_id", how="left")
    missing = df[df["model_category"].isna()]
    df = df[df["model_category"].notna()].copy()
    df["correct"] = df["model_category"] == df["my_category"]

    print(f"File: {path.name}  |  model {outputs['model'].iat[0]}, prompt {outputs['prompt_version'].iat[0]}, split {args.split}")
    print(f"Companies in split: {len(sample)}, classified: {len(df)}, missing: {len(missing)}")
    if len(missing):
        print("  Missing: " + ", ".join(missing["name"]))

    acc = accuracy_table(df)
    print("\nAccuracy (95% Wilson interval):")
    for _, r in acc.iterrows():
        if r["n"]:
            print(f"  {r['group']:17} {r['value']:7} {r['correct']:3}/{r['n']:<3} = {r['accuracy']:6.1%}"
                  f"   [{r['ci_low']:.0%}, {r['ci_high']:.0%}]")
        else:
            print(f"  {r['group']:17} {r['value']:7}   0 companies")

    unknown = (~df["model_knows_company"].astype(bool)).sum()
    print(f"\nknows_company = false: {unknown} of {len(df)} ({unknown / len(df):.0%})")
    print(f"Needed a retry: {(df['attempts'] > 1).sum()} of {len(df)}")

    confusion = pd.crosstab(
        pd.Categorical(df["my_category"], categories=CATEGORIES, ordered=True),
        pd.Categorical(df["model_category"], categories=CATEGORIES, ordered=True),
        rownames=["mine"], colnames=["model"], dropna=False,
    )
    print("\nConfusion matrix (rows = my label, columns = model):")
    print(confusion.to_string())

    errors = df[~df["correct"]][["company_id", "name", "country", "size_group", "my_category", "my_confidence",
                                 "model_category", "model_confidence", "model_knows_company",
                                 "model_rationale", "my_note"]]
    print(f"\nErrors ({len(errors)}):")
    for _, e in errors.iterrows():
        print(f"\n  {e['name']} ({e['country']}, {e['size_group']})")
        print(f"    mine:  {e['my_category']} ({e['my_confidence']})")
        print(f"    model: {e['model_category']} ({e['model_confidence']}, knows_company={e['model_knows_company']})")
        print(f"    model rationale: {e['model_rationale']}")
        print(f"    my note:         {e['my_note'] or '(none)'}")

    prefix = RESULTS_DIR / f"{path.stem}_eval_{args.split}"
    acc.to_csv(f"{prefix}_accuracy.csv", index=False)
    confusion.to_csv(f"{prefix}_confusion.csv")
    errors.to_csv(f"{prefix}_errors.csv", index=False)
    print(f"\nSaved {prefix.name}_accuracy.csv, _confusion.csv, _errors.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    main()

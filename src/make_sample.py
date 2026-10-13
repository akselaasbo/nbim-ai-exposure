"""Draw the hand-labelling sample from data/holdings_equities.csv.

20 companies from the 500 largest by market value and 20 from the rest. Within each
size group, 10 go to dev and 10 to test. Seed 42. No API calls.

Refuses to overwrite an existing sample.csv (it holds hand labels) unless --force is given.
"""

import argparse
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT, SEED

HOLDINGS_FILE = PROJECT_ROOT / "data" / "holdings_equities.csv"
SAMPLE_FILE = PROJECT_ROOT / "data" / "labels" / "sample.csv"

TOP_N = 500
PER_GROUP = 20
DEV_PER_GROUP = 10


def draw_sample(holdings: pd.DataFrame) -> pd.DataFrame:
    # Sort by value; company_id breaks ties so the order is fully deterministic.
    ranked = holdings.sort_values(["market_value_nok", "company_id"], ascending=[False, True])
    ranked["size_group"] = ["top500"] * TOP_N + ["rest"] * (len(ranked) - TOP_N)

    parts = []
    for group in ["top500", "rest"]:
        # sample() returns rows in random order, so the first 10 are a random half.
        drawn = ranked[ranked["size_group"] == group].sample(n=PER_GROUP, random_state=SEED)
        drawn["split"] = ["dev"] * DEV_PER_GROUP + ["test"] * (PER_GROUP - DEV_PER_GROUP)
        parts.append(drawn)

    sample = pd.concat(parts)
    sample = sample[["company_id", "name", "country", "industry", "size_group", "split"]]
    for col in ["my_category", "my_confidence", "my_note"]:
        sample[col] = ""
    return sample.sort_values(["split", "size_group", "company_id"]).reset_index(drop=True)


def print_summary(sample: pd.DataFrame) -> None:
    print("Per size_group and split:")
    print(pd.crosstab(sample["size_group"], sample["split"], margins=True).to_string())
    print("\nPer country:")
    print(sample["country"].value_counts().to_string())
    print("\nPer industry:")
    print(sample["industry"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the hand-labelling sample.")
    parser.add_argument("--output", default=SAMPLE_FILE)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing sample file.")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"{output} already exists and may contain hand labels. Use --force to overwrite.")

    holdings = pd.read_csv(HOLDINGS_FILE)
    sample = draw_sample(holdings)
    sample.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(sample)} companies to {output}\n")
    print_summary(sample)


if __name__ == "__main__":
    main()

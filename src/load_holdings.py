"""Read NBIM's equity holdings file and write a clean data/holdings_equities.csv.

The raw file is UTF-16 with a BOM, separated by ';'. It has no company identifier,
so company_id is assigned from the row order in the raw file (NBIM00001, ...).
"""

import argparse

import pandas as pd

from config import PROJECT_ROOT

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "eq_20260630.csv"
OUT_FILE = PROJECT_ROOT / "data" / "holdings_equities.csv"

COLUMNS = {
    "Name": "name",
    "Country": "country",
    "Industry": "industry",
    "Market Value(NOK)": "market_value_nok",
    "Market Value(USD)": "market_value_usd",
}


def load_raw(path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", encoding="utf-16")
    df = raw[list(COLUMNS)].rename(columns=COLUMNS)
    df["name"] = df["name"].str.strip()
    df.insert(0, "company_id", [f"NBIM{i:05d}" for i in range(1, len(df) + 1)])
    return df


def print_summary(df: pd.DataFrame) -> None:
    total_nok = df["market_value_nok"].sum()
    by_value = df.sort_values("market_value_nok", ascending=False)

    print(f"Companies:          {len(df):,}")
    print(f"Total value (NOK):  {total_nok:,.0f}  ({total_nok / 1e12:.2f} trillion)")

    print("\nTop 10 by market value (NOK):")
    for _, row in by_value.head(10).iterrows():
        share = row["market_value_nok"] / total_nok
        print(f"  {row['name'][:40]:40} {row['country'][:15]:15} {row['market_value_nok'] / 1e9:8.1f} bn  {share:6.2%}")

    top500_share = by_value.head(500)["market_value_nok"].sum() / total_nok
    print(f"\nShare of value in the 500 largest: {top500_share:.1%}")

    dup_mask = df.duplicated(subset=["name", "country"], keep=False)
    print(f"\nDuplicates on name+country: {dup_mask.sum()} rows")
    if dup_mask.any():
        print(df[dup_mask].sort_values(["name", "country"]).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean NBIM equity holdings.")
    parser.add_argument("--input", default=RAW_FILE)
    parser.add_argument("--output", default=OUT_FILE)
    args = parser.parse_args()

    df = load_raw(args.input)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {args.output}\n")
    print_summary(df)


if __name__ == "__main__":
    main()

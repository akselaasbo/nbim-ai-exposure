"""Read hand labels from sample_labelling.xlsx back into data/labels/sample.csv.

Only the label columns (my_category, my_confidence, my_note) are taken from Excel.
They are matched on company_id, so sorting rows in Excel is harmless, and the other
columns in sample.csv cannot be changed by accident.

Checks:
- the Excel file has exactly the same company_ids as sample.csv
- every non-empty my_category / my_confidence is an allowed value (else nothing is written)
- lists rows where my_category or my_confidence is still empty (these are written as empty)

A timestamped backup of the old sample.csv is saved in data/labels/backups/ first.
"""

import argparse
import shutil
from datetime import datetime

import pandas as pd

from config import PROJECT_ROOT
from prompt import CATEGORIES, CONFIDENCE_LEVELS

LABELS_DIR = PROJECT_ROOT / "data" / "labels"
SAMPLE_FILE = LABELS_DIR / "sample.csv"
XLSX_FILE = LABELS_DIR / "sample_labelling.xlsx"
BACKUP_DIR = LABELS_DIR / "backups"

LABEL_COLUMNS = ["my_category", "my_confidence", "my_note"]
ALLOWED = {"my_category": CATEGORIES, "my_confidence": CONFIDENCE_LEVELS}


def read_labels(path) -> pd.DataFrame:
    labels = pd.read_excel(path, dtype=str).fillna("")
    labels = labels[["company_id"] + LABEL_COLUMNS]
    return labels.apply(lambda col: col.str.strip())


def check_ids(sample: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    errors = []
    duplicated = labels.loc[labels["company_id"].duplicated(), "company_id"].tolist()
    if duplicated:
        errors.append(f"Duplicate company_id in Excel: {duplicated}")
    only_in_excel = sorted(set(labels["company_id"]) - set(sample["company_id"]))
    only_in_csv = sorted(set(sample["company_id"]) - set(labels["company_id"]))
    if only_in_excel:
        errors.append(f"company_id in Excel but not in sample.csv: {only_in_excel}")
    if only_in_csv:
        errors.append(f"company_id in sample.csv but not in Excel: {only_in_csv}")
    return errors


def check_values(merged: pd.DataFrame) -> list[str]:
    errors = []
    for column, allowed in ALLOWED.items():
        filled = merged[merged[column] != ""]
        for _, row in filled[~filled[column].isin(allowed)].iterrows():
            errors.append(f"{row['company_id']} {row['name']}: {column} = {row[column]!r} (allowed: {', '.join(allowed)})")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Import hand labels from Excel into sample.csv.")
    parser.add_argument("--input", default=XLSX_FILE)
    args = parser.parse_args()

    sample = pd.read_csv(SAMPLE_FILE, dtype=str, keep_default_na=False)
    labels = read_labels(args.input)

    errors = check_ids(sample, labels)
    if errors:
        raise SystemExit("Not written. " + "\n".join(errors))

    merged = sample.drop(columns=LABEL_COLUMNS).merge(labels, on="company_id", how="left")
    errors = check_values(merged)
    if errors:
        raise SystemExit("Not written. Invalid values:\n  " + "\n  ".join(errors))

    BACKUP_DIR.mkdir(exist_ok=True)
    backup = BACKUP_DIR / f"sample_{datetime.now():%Y%m%d_%H%M%S}.csv"
    shutil.copy2(SAMPLE_FILE, backup)
    merged.to_csv(SAMPLE_FILE, index=False, encoding="utf-8")
    print(f"Backup:  {backup}")
    print(f"Wrote:   {SAMPLE_FILE}")

    missing = merged[(merged["my_category"] == "") | (merged["my_confidence"] == "")]
    print(f"\nLabelled: {len(merged) - len(missing)} of {len(merged)}")
    print(pd.crosstab(merged["split"], merged["my_category"].replace("", "(empty)")).to_string())
    if len(missing):
        print(f"\nMissing my_category or my_confidence ({len(missing)} rows):")
        print(missing[["company_id", "name", "split", "my_category", "my_confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()

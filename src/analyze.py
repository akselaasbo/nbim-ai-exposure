"""Summarise the full-fund classification: results/summary.md, spot_check.csv and two figures.

Joins the valid answers in results/claude-haiku-5-5_fund_v1.jsonl with data/holdings_equities.csv.
Shares of "portfolio value" are shares of the total NOK market value of the equity holdings.
Figure 2 uses the labelled test split (results/claude-haiku-5-5_test_v1_eval_test_accuracy.csv),
not the full run.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import PROJECT_ROOT, SEED

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
HOLDINGS_FILE = PROJECT_ROOT / "data" / "holdings_equities.csv"
FUND_FILE = RESULTS_DIR / "claude-haiku-5-5_fund_v1.jsonl"
TEST_ACCURACY_FILE = RESULTS_DIR / "claude-haiku-5-5_test_v1_eval_test_accuracy.csv"

TOP_N = 500
CATEGORIES = ["ENABLER", "BENEFICIARY", "DISRUPTED", "LIMITED"]
CONFIDENCES = ["high", "medium", "low"]
# Same three groups in every table: AI sellers and winners, losers, and the rest.
GROUP_OF = {"ENABLER": "ENABLER + BENEFICIARY", "BENEFICIARY": "ENABLER + BENEFICIARY",
            "DISRUPTED": "DISRUPTED", "LIMITED": "LIMITED"}
GROUPS = ["ENABLER + BENEFICIARY", "DISRUPTED", "LIMITED"]

# Colours from the dataviz reference palette (light mode); LIMITED is neutral so it recedes.
COLORS = {"ENABLER": "#2a78d6", "BENEFICIARY": "#1baf7a", "DISRUPTED": "#eb6834", "LIMITED": "#9a9994"}
INK, INK_MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


# ---------- data ----------

def load_data() -> pd.DataFrame:
    holdings = pd.read_csv(HOLDINGS_FILE, dtype={"company_id": str})
    ranked = holdings.sort_values(["market_value_nok", "company_id"], ascending=[False, True])
    holdings["size_group"] = pd.Series(["top500"] * TOP_N + ["rest"] * (len(ranked) - TOP_N),
                                       index=ranked.index)
    rows = []
    for line in FUND_FILE.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["output"] is not None:
            rows.append({"company_id": record["company_id"], **record["output"]})
    answers = pd.DataFrame(rows).drop_duplicates("company_id", keep="last")
    df = holdings.merge(answers, on="company_id", how="inner", validate="one_to_one")
    if len(df) != len(holdings):
        raise SystemExit(f"{len(holdings) - len(df)} holdings have no valid answer.")
    df["group"] = df["category"].map(GROUP_OF)
    df["share"] = df["market_value_nok"] / df["market_value_nok"].sum()
    return df


# ---------- markdown helpers ----------

def md_table(df: pd.DataFrame) -> str:
    def cell(value) -> str:
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{x:.1%}"


def value_share(df: pd.DataFrame, by: str, order: list[str]) -> pd.Series:
    """Share of the value in df that falls on each value of column `by`."""
    total = df["market_value_nok"].sum()
    return (df.groupby(by)["market_value_nok"].sum() / total).reindex(order).fillna(0.0)


# ---------- sections ----------

def section_headline(df: pd.DataFrame) -> str:
    confident = df[df["confidence"].isin(["high", "medium"])]
    table = pd.DataFrame({
        "Group": GROUPS,
        "All companies": [pct(v) for v in value_share(df, "group", GROUPS)],
        "Only high/medium confidence": [pct(v) for v in value_share(confident, "group", GROUPS)],
    })
    covered = confident["market_value_nok"].sum() / df["market_value_nok"].sum()
    return ("## 0. Share of the equity portfolio's market value\n\n" + md_table(table) +
            f"\n\nThe right-hand column is computed only on companies where the model's confidence is high or medium: "
            f"{len(confident):,} of {len(df):,} companies, {pct(covered)} of the portfolio value. "
            f"Shares in each column sum to 100% of that column's own value.\n")


def cat_cell(part: pd.DataFrame, total_value: float) -> str:
    return f"{pct(part['market_value_nok'].sum() / total_value)} ({len(part):,})"


def section_by_category(df: pd.DataFrame) -> str:
    out = ["## 1. Category: share of market value (number of companies)\n"]
    total_value = df["market_value_nok"].sum()

    rows = [{"Category": c, "Share of value": pct(df[df.category == c]["market_value_nok"].sum() / total_value),
             "Companies": f"{(df.category == c).sum():,}"} for c in CATEGORIES]
    out += ["### Total\n", md_table(pd.DataFrame(rows)), ""]

    def breakdown(by: str, title: str, order: list[str]) -> None:
        rows = []
        for key in order:
            part = df[df[by] == key]
            if part.empty:
                continue
            value = part["market_value_nok"].sum()
            rows.append({by.replace("_", " ").title(): key,
                         "Companies": f"{len(part):,}",
                         "Share of portfolio": pct(value / total_value),
                         **{c: cat_cell(part[part.category == c], value) for c in CATEGORIES}})
        out.extend([f"### {title}\n", "Each category cell is the share of that row's own value (number of companies).\n",
                    md_table(pd.DataFrame(rows)), ""])

    industries = df.groupby("industry")["market_value_nok"].sum().sort_values(ascending=False).index.tolist()
    breakdown("industry", "Per industry", industries)
    breakdown("size_group", "Per size group (500 largest vs. the rest)", ["top500", "rest"])
    return "\n".join(out)


def section_confidence(df: pd.DataFrame) -> str:
    total_value = df["market_value_nok"].sum()
    rows = [{"Confidence": c, "Share of value": pct(df[df.confidence == c]["market_value_nok"].sum() / total_value),
             "Companies": f"{(df.confidence == c).sum():,}"} for c in CONFIDENCES]
    unknown = df[~df["knows_company"]]
    return ("## 2. Confidence and unknown companies\n\n" + md_table(pd.DataFrame(rows)) +
            f"\n\nknows_company = false: {pct(unknown['market_value_nok'].sum() / total_value)} of the value, "
            f"{len(unknown):,} companies ({pct(len(unknown) / len(df))}).\n")


def section_top(df: pd.DataFrame) -> str:
    out = ["## 3. The 15 largest companies per category\n"]
    for category in ["ENABLER", "DISRUPTED", "BENEFICIARY"]:
        top = df[df.category == category].sort_values("market_value_nok", ascending=False).head(15)
        table = pd.DataFrame({"Company": top["name"], "Country": top["country"], "Industry": top["industry"],
                              "Share of portfolio": [f"{s:.2%}" for s in top["share"]],
                              "Confidence": top["confidence"], "Rationale": top["rationale"]})
        out += [f"### {category}\n", md_table(table), ""]
    return "\n".join(out)


def make_spot_check(df: pd.DataFrame) -> pd.DataFrame:
    parts = [df[df.category == c].sample(n=15, random_state=SEED) for c in ["DISRUPTED", "BENEFICIARY"]]
    spot = pd.concat(parts)[["company_id", "category", "name", "country", "industry", "confidence",
                             "knows_company", "rationale"]]
    spot["my_view"] = ""  # to fill in by hand: agree / disagree
    return spot.reset_index(drop=True)


def section_spot_check(spot: pd.DataFrame) -> str:
    out = ["## 4. Random spot check (seed 42): 15 DISRUPTED and 15 BENEFICIARY\n",
           "Also saved as `results/spot_check.csv` with an empty `my_view` column.\n"]
    for category in ["DISRUPTED", "BENEFICIARY"]:
        part = spot[spot.category == category]
        table = part[["name", "country", "industry", "confidence", "rationale"]].rename(columns=str.capitalize)
        out += [f"### {category}\n", md_table(table), ""]
    return "\n".join(out)


# ---------- figures ----------

def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_category_shares(df: pd.DataFrame, path: Path) -> None:
    total_value = df["market_value_nok"].sum()
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    style_axes(ax)
    for i, category in enumerate(CATEGORIES):
        part = df[df.category == category]
        share = part["market_value_nok"].sum() / total_value
        low = part[part.confidence == "low"]["market_value_nok"].sum() / total_value
        ax.bar(i, share - low, width=0.6, color=COLORS[category])  # high + medium confidence
        ax.bar(i, low, bottom=share - low, width=0.6, facecolor=SURFACE, edgecolor=COLORS[category],
               hatch="////", linewidth=1.2)  # low confidence: hatched
        ax.text(i, share + 0.012, f"{share:.1%}", ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
        if low > 0.001:
            ax.text(i, share + 0.055, f"of which low: {low / share:.0%}", ha="center", va="bottom",
                    color=INK_MUTED, fontsize=9)
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, color=INK, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Share of equity portfolio value per AI-exposure category", loc="left", color=INK, fontsize=12)
    ax.text(0, -0.13, f"Hatched part = answers with low model confidence. {len(df):,} companies, Haiku 5.5, codebook v1.",
            transform=ax.transAxes, color=INK_MUTED, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_test_accuracy(path: Path) -> None:
    acc = pd.read_csv(TEST_ACCURACY_FILE)
    acc = acc[acc["group"] == "model_confidence"].set_index("value").reindex(CONFIDENCES)
    fig, ax = plt.subplots(figsize=(6.5, 4.4), facecolor=SURFACE)
    style_axes(ax)
    for i, level in enumerate(CONFIDENCES):
        row = acc.loc[level]
        ax.bar(i, row["accuracy"], width=0.55, color=COLORS["ENABLER"])
        ax.errorbar(i, row["accuracy"], yerr=[[row["accuracy"] - row["ci_low"]], [row["ci_high"] - row["accuracy"]]],
                    color=INK_MUTED, capsize=4, linewidth=1.2)
        ax.text(i, 0.03, f"{int(row['correct'])}/{int(row['n'])}", ha="center", va="bottom", color="white",
                fontsize=11, fontweight="bold")
        ax.text(i, row["ci_high"] + 0.02, f"{row['accuracy']:.0%}", ha="center", va="bottom", color=INK, fontsize=11)
    ax.set_xticks(range(len(CONFIDENCES)), [f"{c} confidence" for c in CONFIDENCES], color=INK, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Haiku 5.5 accuracy on the test split, by stated confidence", loc="left", color=INK, fontsize=12)
    ax.text(0, -0.13, "Bar label = correct/companies. Whiskers = 95% Wilson interval. 20 test companies in total.",
            transform=ax.transAxes, color=INK_MUTED, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    argparse.ArgumentParser(description="Summarise the full-fund classification.").parse_args()

    df = load_data()
    FIGURES_DIR.mkdir(exist_ok=True)
    spot = make_spot_check(df)
    spot.to_csv(RESULTS_DIR / "spot_check.csv", index=False, encoding="utf-8")
    fig_category_shares(df, FIGURES_DIR / "fig1_value_share_by_category.png")
    fig_test_accuracy(FIGURES_DIR / "fig2_test_accuracy_by_confidence.png")

    total_nok = df["market_value_nok"].sum()
    header = (f"# AI exposure of the equity portfolio: summary\n\n"
              f"Haiku 5.5, codebook v1, {len(df):,} companies, total market value {total_nok / 1e12:.2f} trillion NOK. "
              f"Generated by `src/analyze.py`; every number here comes from the model's answers, not from my labels. "
              f"Accuracy against my labels is in the evaluate output (95% on dev and on test, 20 companies each, "
              f"no DISRUPTED company in either set).\n")
    sections = [header, section_headline(df), section_by_category(df), section_confidence(df),
                section_top(df), section_spot_check(spot)]
    (RESULTS_DIR / "summary.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'summary.md'}, spot_check.csv and 2 figures in {FIGURES_DIR}\n")

    print("The 20 largest companies:")
    for _, r in df.sort_values("market_value_nok", ascending=False).head(20).iterrows():
        print(f"\n  {r['name']} ({r['country']}), {r['share']:.2%} of portfolio")
        print(f"    {r['category']} / {r['confidence']} / knows_company={r['knows_company']}")
        print(f"    {r['rationale']}")


if __name__ == "__main__":
    main()

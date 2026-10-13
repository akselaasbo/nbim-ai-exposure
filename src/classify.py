"""Classify sample companies by AI exposure with Claude.

One request per company. System prompt = docs/codebook.md (cached with cache_control).
The answer is forced into OUTPUT_SCHEMA with output_config.format (structured outputs),
which works the same way on claude-haiku-5-5 and claude-sonnet-5-5. The model gets no tools.

Each answer is validated in Python (allowed values, knows_company=false -> confidence=low) and
retried up to 3 times. A rationale over 25 words is only a warning: the answer is accepted and
the line gets word_limit_exceeded=true. Rejected raw answers are kept in rejected_raw.
Every company becomes one JSONL line.
Re-running the same command skips companies that already have a valid line (resume).
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd

from config import DEFAULT_MODEL, PRICES_PER_MTOK, PROJECT_ROOT, SEED, get_client
from prompt import CATEGORIES, CONFIDENCE_LEVELS, OUTPUT_SCHEMA, PROMPT_VERSION, build_system_prompt

SAMPLE_FILE = PROJECT_ROOT / "data" / "labels" / "sample.csv"
HOLDINGS_FILE = PROJECT_ROOT / "data" / "holdings_equities.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

EFFORT = "medium"
MAX_TOKENS = 4000  # thinking counts toward max_tokens; the JSON answer itself is short
MAX_RETRIES = 3
SMOKE_N = 20
MAX_RATIONALE_WORDS = 25
EST_OUTPUT_TOKENS = 1000  # conservative guess per request (thinking + JSON), for cost estimates only
USAGE_FIELDS = ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]


# ---------- request building ----------

def build_user_message(company: pd.Series) -> str:
    return f"Company: {company['name']}\nCountry: {company['country']}\nIndustry: {company['industry']}"


def build_params(model: str, system_prompt: str, company: pd.Series) -> dict:
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": build_user_message(company)}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    }


# ---------- validation ----------

def parse_and_validate(message) -> tuple[dict | None, list[str], str | None, bool]:
    """Return (output, errors, raw_text, word_limit_exceeded).

    output is None if the response cannot be used. The word limit never causes an error.
    """
    if message.stop_reason == "refusal":
        return None, [f"refusal: {message.stop_details}"], None, False
    text = next((block.text for block in message.content if block.type == "text"), None)
    if message.stop_reason == "max_tokens":
        return None, ["stopped at max_tokens"], text, False
    if text is None:
        return None, ["no text block in response"], None, False
    try:
        output = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [f"invalid JSON: {e}"], text, False

    errors = []
    if output.get("category") not in CATEGORIES:
        errors.append(f"invalid category: {output.get('category')!r}")
    if output.get("confidence") not in CONFIDENCE_LEVELS:
        errors.append(f"invalid confidence: {output.get('confidence')!r}")
    if not isinstance(output.get("knows_company"), bool):
        errors.append(f"knows_company is not a bool: {output.get('knows_company')!r}")
    if output.get("knows_company") is False and output.get("confidence") != "low":
        errors.append(f"knows_company=false but confidence={output.get('confidence')!r}")
    if not str(output.get("rationale", "")).strip():
        errors.append("empty rationale")
    word_limit_exceeded = len(str(output.get("rationale", "")).split()) > MAX_RATIONALE_WORDS
    return (None if errors else output), errors, text, word_limit_exceeded


def add_usage(total: dict, usage) -> None:
    for field in USAGE_FIELDS:
        total[field] += getattr(usage, field, 0) or 0


def make_line(company_id: str, model: str, output, usage: dict, attempts: int, errors: list[str],
              word_limit_exceeded: bool = False, rejected_raw: list | None = None) -> dict:
    return {
        "company_id": company_id,
        "output": output,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "effort": EFFORT,
        "usage": usage,
        "attempts": attempts,
        "word_limit_exceeded": word_limit_exceeded,  # rationale over 25 words (accepted, warning only)
        "validation_errors": errors,  # errors from rejected attempts
        "rejected_raw": rejected_raw or [],  # raw model text of rejected attempts
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------- one company, synchronous ----------

def classify_one(client, params: dict, company_id: str) -> dict:
    usage = {field: 0 for field in USAGE_FIELDS}
    all_errors, rejected_raw = [], []
    for attempt in range(1, MAX_RETRIES + 2):  # first try + 3 retries
        try:
            message = client.messages.create(**params)
        except anthropic.BadRequestError:
            raise  # a bug in the request; retrying will not help
        except anthropic.APIError as e:
            all_errors.append(f"attempt {attempt}: API error: {e}")
        else:
            add_usage(usage, message.usage)
            output, errors, raw, too_long = parse_and_validate(message)
            if output is not None:
                return make_line(company_id, params["model"], output, usage, attempt, all_errors,
                                 too_long, rejected_raw)
            all_errors += [f"attempt {attempt}: {err}" for err in errors]
            rejected_raw.append(raw)
        if attempt <= MAX_RETRIES:
            time.sleep(2 ** attempt)
    return make_line(company_id, params["model"], None, usage, MAX_RETRIES + 1, all_errors,
                     rejected_raw=rejected_raw)


# ---------- files, resume and cost ----------

def load_done(path: Path, model: str) -> set[str]:
    """company_ids that already have a valid output. Refuses to mix models or prompt versions."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["model"] != model or record["prompt_version"] != PROMPT_VERSION:
            raise SystemExit(
                f"{path} has lines from {record['model']} / {record['prompt_version']}; "
                f"this run is {model} / {PROMPT_VERSION}. Use another --out."
            )
        if record["output"] is not None:
            done.add(record["company_id"])
    return done


def append_line(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def measured_output_tokens(model: str) -> float | None:
    """Average output tokens per request in earlier runs of this model (None if no runs)."""
    tokens, lines = 0, 0
    for path in RESULTS_DIR.glob(f"{model}_*_{PROMPT_VERSION}.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            tokens += json.loads(line)["usage"]["output_tokens"]
            lines += 1
    return tokens / lines if lines else None


def estimate_cost(client, params: dict, n: int, batch: bool) -> float:
    """Print a likely and a conservative estimate; return the conservative one."""
    count_params = {k: v for k, v in params.items() if k != "max_tokens"}
    input_tokens = client.messages.count_tokens(**count_params).input_tokens
    price_in, price_out = PRICES_PER_MTOK[params["model"]]
    discount = 0.5 if batch else 1.0
    label = " at batch prices" if batch else ""

    def cost(output_tokens: float) -> float:
        return discount * n * (input_tokens * price_in + output_tokens * price_out) / 1e6

    measured = measured_output_tokens(params["model"])
    print(f"Estimate for {n} requests{label}, {input_tokens} input tokens each, without cache savings:")
    if measured:
        print(f"  likely:        ~{measured:.0f} output tokens each (measured on earlier runs) = USD {cost(measured):.2f}")
    print(f"  conservative:  ~{EST_OUTPUT_TOKENS} output tokens each = USD {cost(EST_OUTPUT_TOKENS):.2f}")
    return cost(EST_OUTPUT_TOKENS)


ASSUME_YES = False  # set by --yes, after the user has approved the printed estimate


def confirm(question: str) -> bool:
    if ASSUME_YES:
        return True
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes", "j", "ja")


def print_run_summary(path: Path, model: str) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ok = [r for r in records if r["output"] is not None]
    totals = {field: sum(r["usage"][field] for r in records) for field in USAGE_FIELDS}
    price_in, price_out = PRICES_PER_MTOK[model]
    print(f"\n{path.name}: {len(ok)} valid of {len(records)} lines")
    print(f"Retried after a validation/API error: {sum(r['attempts'] > 1 for r in ok)}")
    print(f"Accepted answers over the word limit: {sum(r.get('word_limit_exceeded', False) for r in ok)}")
    print("Tokens: " + ", ".join(f"{k}={v:,}" for k, v in totals.items()))
    if totals["cache_read_input_tokens"]:
        print(f"Cache hits: yes, {sum(r['usage']['cache_read_input_tokens'] > 0 for r in records)} of {len(records)} lines read from cache")
    else:
        print("Cache hits: none (cache_read_input_tokens = 0 on every line)")
    approx = (totals["input_tokens"] * price_in + totals["output_tokens"] * price_out
              + totals["cache_creation_input_tokens"] * price_in * 1.25
              + totals["cache_read_input_tokens"] * price_in * 0.1) / 1e6
    print(f"Approximate cost at standard prices: USD {approx:.4f}")


# ---------- batch mode ----------

def submit_batch(client, todo: pd.DataFrame, params_by_id: dict, out: Path) -> str:
    """Create the batch and remember its id next to the output file."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params_by_id[cid]))
                for cid in todo["company_id"]]
    batch = client.messages.batches.create(requests=requests)
    out.with_suffix(".batch_id").write_text(batch.id, encoding="utf-8")
    return batch.id


def wait_for_batch(client, batch_id: str) -> None:
    batch = client.messages.batches.retrieve(batch_id)
    while batch.processing_status != "ended":
        print(f"  {batch.processing_status}: {batch.request_counts}")
        time.sleep(30)
        batch = client.messages.batches.retrieve(batch_id)


def fetch_batch(client, batch_id: str, out: Path, model: str) -> None:
    """Append the batch results to out in the same format as a normal run."""
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        raise SystemExit(f"Batch {batch_id} is not finished: {batch.processing_status}, {batch.request_counts}")
    done = load_done(out, model)
    # Results arrive in any order; key by custom_id. Failed ones are written with output=None,
    # so a normal (non-batch) re-run retries them.
    for result in client.messages.batches.results(batch_id):
        if result.custom_id in done:
            continue
        usage = {field: 0 for field in USAGE_FIELDS}
        if result.result.type == "succeeded":
            add_usage(usage, result.result.message.usage)
            output, errors, raw, too_long = parse_and_validate(result.result.message)
            rejected = [raw] if output is None else []
        else:
            output, errors, too_long, rejected = None, [f"batch result: {result.result.type}"], False, []
        append_line(out, make_line(result.custom_id, model, output, usage, 1, errors, too_long, rejected))


# ---------- main ----------

def load_companies(split: str, limit: int | None) -> pd.DataFrame:
    """dev/test/all come from sample.csv. smoke = random companies NOT in the sample (seed 42).
    fund = every holding."""
    sample = pd.read_csv(SAMPLE_FILE, dtype=str, keep_default_na=False)
    if split in ("smoke", "fund"):
        holdings = pd.read_csv(HOLDINGS_FILE, dtype=str, keep_default_na=False)
        if split == "fund":
            return holdings
        outside = holdings[~holdings["company_id"].isin(sample["company_id"])]
        return outside.sample(n=limit or SMOKE_N, random_state=SEED)
    return sample if split == "all" else sample[sample["split"] == split]


def main() -> None:
    global ASSUME_YES
    parser = argparse.ArgumentParser(description="Classify companies by AI exposure.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PRICES_PER_MTOK))
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all", "smoke", "fund"],
                        help="dev/test/all = labelled sample; smoke = 20 random companies outside the sample; "
                             "fund = all holdings.")
    parser.add_argument("--limit", type=int, help="Only the first N companies not yet done (smoke: N random companies).")
    parser.add_argument("--out", help="Output JSONL (default: results/<model>_<split>_<prompt_version>.jsonl).")
    parser.add_argument("--batch", action="store_true", help="Use the Message Batches API (50%% price, async).")
    parser.add_argument("--no-wait", action="store_true", help="With --batch: submit, print the batch id and stop.")
    parser.add_argument("--fetch", metavar="BATCH_ID", help="Append the results of a finished batch to --out and stop.")
    parser.add_argument("--estimate-only", action="store_true", help="Print the cost estimate and stop.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompts (use after approving the estimate).")
    args = parser.parse_args()
    ASSUME_YES = args.yes

    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.model}_{args.split}_{PROMPT_VERSION}.jsonl"

    if args.fetch:
        fetch_batch(get_client(), args.fetch, out, args.model)
        print_run_summary(out, args.model)
        return

    companies = load_companies(args.split, args.limit)
    done = load_done(out, args.model)
    todo = companies[~companies["company_id"].isin(done)]
    if args.limit and args.split != "smoke":
        todo = todo.head(args.limit)
    print(f"Model {args.model}, prompt {PROMPT_VERSION}, effort {EFFORT}, split {args.split}")
    print(f"{len(companies)} companies, {len(done)} already done, {len(todo)} to run -> {out}")
    if todo.empty:
        if out.exists():
            print_run_summary(out, args.model)
        return

    if args.split in ("test", "all") and not args.estimate_only and not confirm(
            "This run includes the TEST split, which is used once. Continue?"):
        raise SystemExit("Stopped.")

    client = get_client()
    system_prompt = build_system_prompt()
    params_by_id = {row["company_id"]: build_params(args.model, system_prompt, row) for _, row in todo.iterrows()}

    cost = estimate_cost(client, next(iter(params_by_id.values())), len(todo), args.batch)
    if args.estimate_only:
        return
    if (args.batch or cost > 2) and not confirm("Start?"):
        raise SystemExit("Stopped.")

    if args.batch:
        batch_id = submit_batch(client, todo, params_by_id, out)
        print(f"Batch {batch_id} submitted with {len(todo)} requests (id saved in {out.with_suffix('.batch_id')}).")
        if args.no_wait:
            return
        wait_for_batch(client, batch_id)
        fetch_batch(client, batch_id, out, args.model)
    else:
        for i, (_, company) in enumerate(todo.iterrows(), start=1):
            record = classify_one(client, params_by_id[company["company_id"]], company["company_id"])
            append_line(out, record)
            result = record["output"]["category"] if record["output"] else "FAILED"
            print(f"  [{i}/{len(todo)}] {company['name'][:40]:40} {result:12} "
                  f"attempts={record['attempts']} cache_read={record['usage']['cache_read_input_tokens']}")

    print_run_summary(out, args.model)


if __name__ == "__main__":
    main()

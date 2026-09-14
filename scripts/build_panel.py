"""Build the firm-month panel from SEC, price and FRED data.

Resumable: each firm's rows are written to a per-firm parquet shard as soon as
they are built, so an interrupted run picks up where it left off and a rebuild
costs nothing for firms already done.

    python scripts/build_panel.py --firms 700 --start 2015-01-31 --workers 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merton.data import panel as panel_mod
from merton.data import symbols

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SHARD_DIR = ROOT / ".cache" / "shards"

log = logging.getLogger("build_panel")


def month_ends(start: str, end: str | None = None) -> pd.DatetimeIndex:
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(start=start, end=end, freq="ME")


def _load_profile(path: Path) -> dict:
    return pd.read_json(path, typ="series", convert_dates=["default_date"]).to_dict()


def build_one(cik: int, ticker, start: str, dates, skip_financials: bool):
    """Fetch and assemble one firm. Returns a (status, profile) pair.

    Every outcome is a status rather than an exception, so the run can report
    *why* firms dropped out. A silent dropna at the end would hide whether a
    firm was lost to a missing ticker or a missing balance sheet, and those two
    have very different implications for bias in the surviving sample.
    """
    shard = SHARD_DIR / f"{cik:010d}.parquet"
    profile_path = SHARD_DIR / f"{cik:010d}.profile.json"

    if shard.exists() and profile_path.exists():
        return "cached", _load_profile(profile_path)
    if profile_path.exists() and not shard.exists():
        return "cached_empty", _load_profile(profile_path)

    profile = panel_mod.firm_profile(cik, start, skip_financials=skip_financials)
    if profile is None:
        return "no_profile", None
    if profile.get("excluded") == "financial":
        return "financial", None

    if not isinstance(ticker, str) or not ticker:
        # Not in SEC's ticker file: almost always a delisted firm, and those
        # are exactly the observations a default study must not lose.
        ticker, score = symbols.search_symbol(profile["name"])
        if ticker is None:
            return "no_ticker", None
        profile["resolved_by"] = f"name-search({score:.2f})"
    profile["ticker"] = ticker

    rows = panel_mod.firm_rows(profile, ticker, dates)
    pd.Series(profile).to_json(profile_path)
    if rows.empty:
        return "no_rows", profile
    rows.to_parquet(shard, index=False)
    return "built", profile


def build(n_firms: int, start: str, skip_financials: bool = True,
          workers: int = 6) -> pd.DataFrame:
    """Walk the candidate pool until ``n_firms`` firms have usable rows.

    Wall-clock here is almost entirely request latency, so candidates are
    worked in parallel. The rate limit still binds globally -- the shared
    session spaces requests per host -- so concurrency raises throughput
    without raising the load any single source sees.
    """
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    dates = month_ends(start)
    log.info("panel window: %s .. %s (%d month-ends)",
             dates[0].date(), dates[-1].date(), len(dates))

    universe = panel_mod.select_universe(max(n_firms * 4, 2000))
    log.info("candidate pool: %d filers", len(universe))
    universe.to_csv(DATA_DIR / "universe.csv", index=False)

    profiles: list[dict] = []
    counts: Counter = Counter()
    lock = threading.Lock()
    stop = threading.Event()
    state = {"built": len(list(SHARD_DIR.glob("*.parquet"))), "done": 0}
    started = time.time()

    def worker(item):
        if stop.is_set():
            return None
        _, row = item
        try:
            status, profile = build_one(int(row["cik"]), row["ticker"], start,
                                        dates, skip_financials)
        except Exception:
            log.exception("failed on CIK %s", row["cik"])
            return ("error", None)

        with lock:
            counts[status] += 1
            state["done"] += 1
            if status in ("built", "cached"):
                state["built"] += 1
                if state["built"] >= n_firms:
                    stop.set()
            if state["done"] % 50 == 0:
                log.info("cand %d/%d | built %d/%d | %.2f cand/s | %s",
                         state["done"], len(universe), state["built"], n_firms,
                         state["done"] / max(time.time() - started, 1e-9), dict(counts))
        return (status, profile)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(worker, universe.iterrows()):
            if result and result[1]:
                profiles.append(result[1])

    log.info("candidate outcomes: %s", dict(counts))
    return assemble(profiles)


def assemble(profiles: list[dict] | None = None) -> pd.DataFrame:
    """Concatenate the shards and attach labels, ratios and rates."""
    shards = sorted(SHARD_DIR.glob("*.parquet"))
    log.info("assembling %d firm shards", len(shards))
    if not shards:
        raise RuntimeError("no firm data was built")

    frame = pd.concat((pd.read_parquet(s) for s in shards), ignore_index=True)
    frame = panel_mod.add_labels(frame)
    frame = panel_mod.add_accounting_ratios(frame)
    frame = panel_mod.attach_rates(frame)
    frame = frame.sort_values(["cik", "date"]).reset_index(drop=True)

    if not profiles:
        profiles = [_load_profile(p) for p in SHARD_DIR.glob("*.profile.json")]
    if profiles:
        prof = pd.DataFrame(profiles).drop_duplicates("cik")
        prof.to_csv(DATA_DIR / "firm_profiles.csv", index=False)
        defaults = prof[prof["default_date"].notna()]
        defaults.to_csv(DATA_DIR / "defaults.csv", index=False)
        log.info("bankruptcy filers among processed firms: %d", len(defaults))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firms", type=int, default=700)
    parser.add_argument("--start", default=panel_mod.DEFAULT_START)
    parser.add_argument("--workers", type=int, default=6,
                        help="parallel fetch workers; the per-host rate limit still binds")
    parser.add_argument("--assemble-only", action="store_true",
                        help="rebuild the panel from existing shards without fetching")
    parser.add_argument("--out", default=str(DATA_DIR / "panel.parquet"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    DATA_DIR.mkdir(exist_ok=True)

    frame = assemble() if args.assemble_only else build(
        args.firms, args.start, workers=args.workers)
    frame.to_parquet(args.out, index=False)
    log.info("wrote %s: %d rows, %d firms, %d default-flagged rows",
             args.out, len(frame), frame["cik"].nunique(),
             int(frame["default_12m"].sum()))


if __name__ == "__main__":
    main()

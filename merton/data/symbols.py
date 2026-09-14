"""Resolve an SEC registrant to a tradable ticker.

Most firms are in SEC's ticker file, but firms that were delisted -- which is
to say, disproportionately the firms that defaulted -- are not. Dropping them
would rebuild exactly the survivorship bias this panel exists to avoid, so
names missing from the ticker file are resolved by searching the price
provider and accepting a match only when the company names genuinely agree.

The match is deliberately conservative. A wrong ticker would attach one firm's
price history to another firm's balance sheet, which is far worse than a miss,
so anything below the similarity threshold is rejected and the firm is dropped.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from difflib import SequenceMatcher

from merton.data.http import session

log = logging.getLogger(__name__)

SEARCH_URL = "https://stockanalysis.com/api/search?q={query}"
# Deliberately severe. At 0.72 this matched ENERPLUS to Energous (WATT),
# SOUTHWESTERN ENERGY to NorthWestern (NWE), RED HAT to Red Cat (RCAT) and
# HAWAIIAN HOLDINGS to First Hawaiian (FHB) -- each of which stapled one firm's
# balance sheet to another firm's share price and produced a row that looked
# entirely reasonable. A miss costs one firm; a false match corrupts the panel
# silently, so the threshold sits where only near-identical names survive.
MATCH_THRESHOLD = 0.95

# Legal-form and filer-index noise that carries no identifying information.
_NOISE = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|COMPANIES|HOLDINGS?|HOLDING|"
    r"GROUP|LTD|LIMITED|PLC|LP|LLC|LLP|TRUST|THE|NEW|CLASS|COM|ADR|SA|NV|AG)\b"
)
_FILER_SUFFIX = re.compile(r"/[A-Z]{2,4}/?$|\(.*?\)")


def normalize_name(name: str) -> str:
    """Strip legal form and EDGAR filer decoration down to the trading name."""
    text = _FILER_SUFFIX.sub(" ", (name or "").upper())
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    text = _NOISE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# A prefix match is only evidence of identity when almost nothing is left
# over. "YELLOW" is a prefix of "YELLOW CAKE", and treating that as a match
# would hand one firm's price history to another.
PREFIX_COVERAGE = 0.8


def name_similarity(a: str, b: str) -> float:
    """Similarity of two company names after normalization.

    A prefix rule supplements the sequence ratio so that "WEWORK" scores full
    marks against "WEWORK INC", but it applies only when the shorter name
    covers most of the longer one -- otherwise every firm whose name starts
    with a common word would match every other.
    """
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    ratio = SequenceMatcher(None, na, nb).ratio()
    if na.startswith(nb) or nb.startswith(na):
        coverage = min(len(na), len(nb)) / max(len(na), len(nb))
        if coverage >= PREFIX_COVERAGE:
            ratio = max(ratio, 0.9)
    return ratio


def search_symbol(name: str, threshold: float = MATCH_THRESHOLD) -> tuple[str | None, float]:
    """Best-matching US-listed symbol for a company name, or (None, score).

    Results carrying an exchange prefix ("otc/YELLQ", "fra/BED") are skipped:
    the provider has no daily history behind those paths, and a foreign listing
    of a similarly named company is precisely the wrong answer.
    """
    query = normalize_name(name) or name
    body = session().get(SEARCH_URL.format(query=urllib.parse.quote(query)))
    if not body or not body.get("data"):
        return None, 0.0

    best, best_score = None, 0.0
    for hit in body["data"]:
        symbol = (hit.get("s") or "").strip()
        if not symbol or "/" in symbol:
            continue
        score = name_similarity(name, hit.get("n", ""))
        if score > best_score:
            best, best_score = symbol.upper(), score

    if best is None or best_score < threshold:
        return None, best_score
    return best, best_score

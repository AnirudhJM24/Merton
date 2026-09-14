"""A polite, cached HTTP session.

Every source used here is a free public endpoint, so the fetcher is built to be
a good citizen: a fixed minimum interval between requests to the same host, a
descriptive User-Agent with a contact address (SEC requires this), bounded
retries with exponential backoff, and an on-disk cache so a re-run costs no
requests at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("MERTON_CACHE", Path(__file__).resolve().parents[2] / ".cache"))

# SEC asks for a descriptive UA with a contact address and caps traffic at
# 10 requests/second; everything here stays well inside that.
SEC_UA = os.environ.get("MERTON_CONTACT", "Merton credit research (jmanirudh308@gmail.com)")
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

MIN_INTERVAL = {           # seconds between consecutive hits on a host
    "data.sec.gov": 0.12,
    "www.sec.gov": 0.12,
    "stockanalysis.com": 0.6,
    "fred.stlouisfed.org": 0.5,
}
DEFAULT_INTERVAL = 0.5


# Transport backend. ``requests`` is the default and is all most people need.
# Some corporate and sandboxed networks route egress through a proxy that
# refuses or stalls urllib3's CONNECT while curl negotiates it fine, so curl is
# available as a drop-in backend: set MERTON_HTTP_BACKEND=curl to force it, or
# leave it on "auto" to fall back automatically after repeated transport
# failures. Both backends return the same (status, content_type, body) triple.
BACKEND = os.environ.get("MERTON_HTTP_BACKEND", "auto").lower()
_TRANSPORT_FAILURES_BEFORE_FALLBACK = 3


class TransportError(RuntimeError):
    """The request never reached the server (proxy, DNS, TLS, timeout)."""


def _get_requests(sess, url, headers, timeout):
    try:
        resp = sess.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise TransportError(str(exc)) from exc
    return resp.status_code, resp.headers.get("Content-Type", ""), resp.text


def _get_curl(url, headers, timeout):
    if not shutil.which("curl"):
        raise TransportError("curl backend requested but curl is not installed")
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        body_path = fh.name
    try:
        cmd = ["curl", "-sS", "--compressed", "--http1.1", "--max-time", str(int(timeout)),
               "-o", body_path, "-w", "%{http_code}\t%{content_type}"]
        for key, value in headers.items():
            cmd += ["-H", f"{key}: {value}"]
        cmd.append(url)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 15)
        if proc.returncode != 0:
            raise TransportError(f"curl exit {proc.returncode}: {proc.stderr.strip()[:200]}")
        status, _, ctype = proc.stdout.partition("\t")
        body = Path(body_path).read_text(errors="replace")
        return int(status or 0), ctype, body
    except subprocess.TimeoutExpired as exc:
        raise TransportError("curl timed out") from exc
    finally:
        Path(body_path).unlink(missing_ok=True)


class PoliteSession:
    """Rate-limited, retrying, disk-cached GET."""

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_days: float | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = None if ttl_days is None else ttl_days * 86400
        self._session = requests.Session()
        self._last_hit: dict[str, float] = {}
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "errors": 0}
        self._backend = BACKEND if BACKEND in ("requests", "curl") else "requests"
        self._transport_failures = 0

    # -- cache ---------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        host = urlparse(url).netloc.replace(":", "_")
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / host / f"{digest}.json"

    def _read_cache(self, path: Path):
        if not path.exists():
            return None
        if self.ttl is not None and time.time() - path.stat().st_mtime > self.ttl:
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)

    # -- throttle ------------------------------------------------------
    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        interval = MIN_INTERVAL.get(host, DEFAULT_INTERVAL)
        with self._lock:
            elapsed = time.monotonic() - self._last_hit.get(host, 0.0)
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_hit[host] = time.monotonic()

    # -- fetch ---------------------------------------------------------
    def _fetch_once(self, url, headers, timeout=45):
        """One attempt through the active backend, with automatic fallback."""
        if self._backend == "curl":
            return _get_curl(url, headers, timeout)
        try:
            return _get_requests(self._session, url, headers, timeout)
        except TransportError:
            self._transport_failures += 1
            if (BACKEND == "auto"
                    and self._transport_failures >= _TRANSPORT_FAILURES_BEFORE_FALLBACK
                    and shutil.which("curl")):
                log.warning("switching HTTP backend to curl after %d transport failures",
                            self._transport_failures)
                self._backend = "curl"
                return _get_curl(url, headers, timeout)
            raise

    def get(self, url: str, *, sec: bool = False, retries: int = 6):
        """Return the parsed body, or None when the resource genuinely is absent.

        A 404 is an ordinary outcome here: not every filer reports every XBRL
        tag, so a missing concept means "try the next tag", not "fail the run".
        Absences are cached; transport failures never are.
        """
        path = self._cache_path(url)
        cached = self._read_cache(path)
        if cached is not None:
            self.stats["hits"] += 1
            return cached["body"]

        headers = {"User-Agent": SEC_UA if sec else BROWSER_UA,
                   "Accept": "*/*"}
        delay = 2.0
        for attempt in range(retries):
            self._throttle(url)
            try:
                status, ctype, text = self._fetch_once(url, headers)
            except TransportError as exc:
                log.debug("transport error %s (%d/%d): %s", url, attempt + 1, retries, exc)
                time.sleep(delay)
                delay *= 2
                continue

            if status == 200:
                body = self._parse(url, ctype, text)
                if body is None:
                    self.stats["errors"] += 1
                    return None
                self.stats["misses"] += 1
                self._write_cache(path, {"url": url, "body": body, "fetched": time.time()})
                return body
            if status == 404:
                # A real "this filer does not report this tag". Cache it so a
                # rebuild does not re-ask for thousands of known-absent tags.
                self._write_cache(path, {"url": url, "body": None, "fetched": time.time()})
                return None
            if status in (403, 429, 500, 502, 503, 504):
                log.debug("status %s for %s, backing off", status, url)
                time.sleep(delay)
                delay *= 2
                continue
            log.debug("unexpected status %s for %s", status, url)
            return None

        self.stats["errors"] += 1
        return None

    @staticmethod
    def _parse(url, ctype, text):
        if "json" in ctype or url.endswith(".json"):
            try:
                return json.loads(text)
            except ValueError:
                return None
        return text


_DEFAULT: PoliteSession | None = None


def session() -> PoliteSession:
    """Process-wide session, so the cache and throttle are shared."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PoliteSession()
    return _DEFAULT

"""Shadow-mode logging: challenger predictions stored next to the champion's (SQLite)."""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    champion_version TEXT NOT NULL,
    challenger_version TEXT NOT NULL,
    champion_score REAL NOT NULL,
    challenger_score REAL NOT NULL,
    features TEXT
);
CREATE INDEX IF NOT EXISTS idx_shadow_ts ON shadow_log(ts);
"""

_STOP = object()


class ShadowLogger:
    """Asynchronous writer: callers enqueue, a background thread batches inserts."""

    def __init__(self, db_path: str, max_queue: int = 100_000):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;" + SCHEMA)
        self._thread = threading.Thread(target=self._run, name="shadow-writer", daemon=True)
        self._thread.start()

    def log(self, request_id, champion_version, challenger_version, champion_scores,
            challenger_scores, features=None) -> None:
        """Enqueue one record per row. Never blocks; drops (and counts) if the queue is full."""
        ts = time.time()
        rows = [
            (ts, request_id, i, champion_version, challenger_version, float(c), float(h),
             None if features is None else json.dumps(features[i]))
            for i, (c, h) in enumerate(zip(champion_scores, challenger_scores))
        ]
        try:
            self._q.put_nowait(rows)
        except queue.Full:
            self.dropped += len(rows)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            batch = []
            consumed = 1
            stop = item is _STOP
            if not stop:
                batch.extend(item)
            # drain whatever else is waiting into one transaction
            while not stop:
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                consumed += 1
                if nxt is _STOP:
                    stop = True
                else:
                    batch.extend(nxt)
            if batch:
                with self._lock, self._conn:
                    self._conn.executemany(
                        "INSERT INTO shadow_log (ts, request_id, row_index, champion_version,"
                        " challenger_version, champion_score, challenger_score, features)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        batch,
                    )
            for _ in range(consumed):
                self._q.task_done()
            if stop:
                return

    def flush(self, timeout: float = 5.0) -> None:
        """Block until everything enqueued so far is written."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._q.unfinished_tasks == 0:
                return
            time.sleep(0.005)

    def close(self) -> None:
        self._q.put(_STOP)
        self._thread.join(timeout=5)
        with self._lock:
            self._conn.close()

    def summary(self, approve_below: float = 0.10, decline_above: float = 0.25) -> dict:
        """Aggregate champion-vs-challenger agreement over everything logged."""
        self.flush()
        with self._lock:
            rows = self._conn.execute(
                "SELECT champion_score, challenger_score FROM shadow_log"
            ).fetchall()
        n = len(rows)
        if n == 0:
            return {"count": 0, "dropped": self.dropped}
        import numpy as np

        a = np.array([r[0] for r in rows])
        b = np.array([r[1] for r in rows])
        d = b - a

        def band(p):
            return np.where(p < approve_below, 0, np.where(p > decline_above, 2, 1))

        return {
            "count": n,
            "dropped": self.dropped,
            "mean_champion": float(a.mean()),
            "mean_challenger": float(b.mean()),
            "mean_abs_diff": float(np.abs(d).mean()),
            "max_abs_diff": float(np.abs(d).max()),
            "p99_abs_diff": float(np.quantile(np.abs(d), 0.99)),
            "correlation": float(np.corrcoef(a, b)[0, 1]) if n > 1 and a.std() > 0 and b.std() > 0 else None,
            "decision_disagreement_rate": float((band(a) != band(b)).mean()),
        }


def main() -> None:
    import argparse

    from .config import Settings

    s = Settings.from_env()
    ap = argparse.ArgumentParser(description="Summarise champion vs challenger shadow logs")
    ap.add_argument("--db", default=s.shadow_db)
    args = ap.parse_args()
    log = ShadowLogger(args.db)
    print(json.dumps(log.summary(s.approve_below, s.decline_above), indent=2))
    log.close()


if __name__ == "__main__":
    main()

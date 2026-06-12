from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        seen = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp, path)


class RunLock:
    def __init__(self, path: Path, token: str, force: bool = False):
        self.path = path
        self.token = token
        self.force = force

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.force and self.path.exists():
            self.path.unlink()
        payload = {"pid": os.getpid(), "token": self.token, "created_at": utc_now()}
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            existing = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"Run lock exists at {self.path}\n{existing}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    def still_owned(self) -> bool:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data.get("token") == self.token
        except Exception:
            return False

    def release(self) -> None:
        if self.still_owned():
            self.path.unlink(missing_ok=True)


def ensure_artifact_dirs(root: Path) -> None:
    for sub in ["cycle_logs", "signals", "snapshots"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def write_cycle_log(root: Path, cycle_index: int, lines: List[str], manifest: Dict[str, Any]) -> None:
    text = [
        f"# Cycle {cycle_index}",
        "",
        f"- timestamp: {utc_now()}",
        f"- counters: `{json.dumps(manifest.get('counters', {}), sort_keys=True)}`",
        "",
        "## Events",
        "",
    ]
    text.extend(f"- {line}" for line in lines)
    atomic_write_text(root / "cycle_logs" / f"cycle_{cycle_index}.md", "\n".join(text) + "\n")


def write_final_report(
    root: Path,
    manifest: Dict[str, Any],
    top_features: List[Dict[str, Any]],
    top_pairs: List[Dict[str, Any]],
    top_clusters: List[Dict[str, Any]],
    top_signals: List[Dict[str, Any]],
) -> None:
    lines = [
        "# Round 5 Signal Mining Report",
        "",
        f"- stop_reason: `{manifest.get('stop_reason')}`",
        f"- days: `{manifest.get('days')}`",
        f"- feature_scores: `{manifest.get('counters', {}).get('feature_scores', 0)}`",
        f"- pair_scores: `{manifest.get('counters', {}).get('pair_scores', 0)}`",
        f"- signals: `{manifest.get('counters', {}).get('signals', 0)}`",
        f"- clusters: `{manifest.get('counters', {}).get('clusters', 0)}`",
        "",
        "## Top Confirmed Signals",
        "",
    ]
    if top_signals:
        for row in top_signals[:20]:
            lines.append(
                f"- `{row.get('signal_id')}` family={row.get('family')} tier={row.get('tier')} "
                f"score={row.get('stability_score')} scope={row.get('scope')}"
            )
    else:
        lines.append("- No confirmed signals found under current thresholds.")

    lines.extend(["", "## Top Feature Scores", ""])
    for row in top_features[:20]:
        lines.append(
            f"- {row.get('family')} `{row.get('product')}` `{row.get('feature')}` h={row.get('horizon')} "
            f"tier={row.get('tier')} score={row.get('stability_score')}"
        )

    lines.extend(["", "## Top Pair / Lead-Lag Scores", ""])
    for row in top_pairs[:20]:
        lines.append(
            f"- {row.get('family')} `{row.get('leader')}` -> `{row.get('follower')}` "
            f"feature={row.get('feature')} h={row.get('horizon')} tier={row.get('tier')} "
            f"score={row.get('stability_score')}"
        )

    lines.extend(["", "## Top Clusters", ""])
    for row in top_clusters[:20]:
        lines.append(
            f"- `{row.get('cluster_id')}` type={row.get('graph_type')} size={row.get('size')} "
            f"score={row.get('cluster_score')} products={row.get('products')}"
        )

    lines.extend(
        [
            "",
            "## Recommended Next Alpha Workstreams",
            "",
            "- Convert only confirmed/promising signal families into sparse candidate strategies.",
            "- Prioritize relationships that hold on both day 3 and day 4 with the same sign.",
            "- Treat unstable high in-sample signals as research notes, not production candidates.",
        ]
    )
    atomic_write_text(root / "final_report.md", "\n".join(lines) + "\n")


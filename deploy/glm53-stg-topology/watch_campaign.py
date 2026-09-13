"""Read-only progress and measured request timestamps for one experiment campaign."""

import argparse
import json
import math
import re
import time
from pathlib import Path

PROFILES = ("pp2", "dpa2", "dpa4", "dpa8")


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def snapshot(root, campaign):
    rows = []
    for profile in PROFILES:
        run = f"{campaign}-{profile}-baseline"
        directory = root / run
        for file in directory.glob("**/requests.jsonl"):
            relative = file.relative_to(directory)
            if (
                file.is_symlink()
                or len(relative.parts) > 3
                or not re.fullmatch(
                    r"r\d+-(short|long-cold|long-warm)", file.parent.name
                )
            ):
                continue
            if file.stat().st_size > 8 * 1024 * 1024:
                continue
            starts, ends, completed, failed, malformed = [], [], 0, 0, 0
            with file.open() as stream:
                for line in stream:
                    try:
                        value = json.loads(line)
                    except ValueError:
                        malformed += 1
                        continue
                    if not isinstance(value, dict) or value.get("measured") is not True:
                        continue
                    if number(value.get("started_at")):
                        starts.append(value["started_at"])
                    if number(value.get("finished_at")):
                        ends.append(value["finished_at"])
                    success = (
                        value.get("status") == 200
                        and value.get("sse_done") is True
                        and not value.get("evidence_errors")
                        and not value.get("proxy_error")
                    )
                    completed += bool(success)
                    failed += not success
            rows.append(
                {
                    "profile": profile,
                    "run_id": run,
                    "stage": str(relative.parent),
                    "job": "glm53-"
                    + run
                    + ("-prep" if relative.parts[0] == "preparation" else ""),
                    "first_measured_started_at": min(starts, default=None),
                    "last_measured_finished_at": max(ends, default=None),
                    "completed_measured": completed,
                    "error_measured": failed,
                    "malformed_records": malformed,
                    "verdict_present": (file.parent / "verdict.json").is_file(),
                }
            )
    return {
        "timestamp": time.time(),
        "scope": "operational_progress_not_performance_qualification",
        "campaign": campaign,
        "workloads": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--root", type=Path, default=Path("/results"))
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,20}", a.campaign):
        p.error("Invalid campaign")
    for _ in range(1440):
        print(json.dumps(snapshot(a.root, a.campaign)), flush=True)
        if a.once:
            break
        time.sleep(30)

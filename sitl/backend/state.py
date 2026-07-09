"""Job store with lightweight disk persistence.

The job dict is intentionally simple (local, single-user), but persisting it
across restarts means the sidebar/history survives `uvicorn --reload` and
crashes. In-flight jobs can't resume (their subprocess is gone), so on load we
mark any queued/processing job as failed rather than lying about their state.
"""
import os
import json
import logging

log = logging.getLogger(__name__)

_STATE_FILE = "outputs/_jobs_state.json"

# The single in-memory job store, shared by main.py and pipeline.py.
jobs: dict = {}


def persist_jobs() -> None:
    """Atomically snapshot the job store to disk. Never raises."""
    try:
        os.makedirs("outputs", exist_ok=True)
        tmp = _STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(jobs, f)
        os.replace(tmp, _STATE_FILE)          # atomic on POSIX
    except Exception as e:                     # persistence must never break a request
        log.warning("Could not persist job state: %s", e)


def load_jobs() -> None:
    """Restore the job store on startup, downgrading interrupted jobs to failed."""
    if not os.path.exists(_STATE_FILE):
        return
    try:
        with open(_STATE_FILE) as f:
            saved = json.load(f)
    except Exception as e:
        log.warning("Could not load job state: %s", e)
        return
    for jid, info in saved.items():
        if info.get("status") in ("queued", "processing"):
            info["status"] = "failed"
            info.setdefault("logs", []).append("Interrupted by server restart.")
        jobs[jid] = info
    log.info("Restored %d job(s) from %s", len(jobs), _STATE_FILE)

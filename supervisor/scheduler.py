"""
Supervisor — Daily task scheduler.

Runs scheduled jobs at specified local times. Currently used for:
- Daily AI news digest at 10:00 MSK (UTC+3 = 07:00 UTC)

The scheduler runs as a daemon thread, checks every 30 seconds if a job
should fire, and enqueues an LLM task when the time comes.
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import threading
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------

class DailyJob:
    """A job that fires once per day at a given UTC hour + minute."""

    def __init__(self, name: str, utc_hour: int, utc_minute: int,
                 handler: Callable[[], None]) -> None:
        self.name = name
        self.utc_hour = utc_hour
        self.utc_minute = utc_minute
        self.handler = handler
        self._last_fired_date: Optional[datetime.date] = None

    def should_fire(self, now_utc: datetime.datetime) -> bool:
        """Return True if this job should fire right now."""
        if now_utc.hour != self.utc_hour:
            return False
        if now_utc.minute != self.utc_minute:
            return False
        today = now_utc.date()
        if self._last_fired_date == today:
            return False
        return True

    def fire(self, now_utc: datetime.datetime) -> None:
        today = now_utc.date()
        self._last_fired_date = today
        try:
            self.handler()
        except Exception as e:
            log.warning("Scheduler job %r failed: %s", self.name, e, exc_info=True)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class DailyScheduler:
    """Daemon thread that runs daily jobs at specified UTC times."""

    def __init__(self, drive_root: pathlib.Path) -> None:
        self._drive_root = drive_root
        self._jobs: List[DailyJob] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_path = drive_root / "state" / "scheduler_state.json"

    def add_job(self, job: DailyJob) -> None:
        """Register a daily job."""
        self._jobs.append(job)

    def start(self) -> None:
        """Start the scheduler daemon thread."""
        self._load_state()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="DailyScheduler")
        self._thread.start()
        log.info("DailyScheduler started with %d job(s)", len(self._jobs))

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        """Check every 30 seconds if any job should fire."""
        while not self._stop_event.is_set():
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            for job in self._jobs:
                if job.should_fire(now_utc):
                    log.info("Firing scheduled job: %s", job.name)
                    job.fire(now_utc)
                    self._save_state()
            self._stop_event.wait(timeout=30)

    # ------------------------------------------------------------------
    # State persistence (survive runtime restarts within the same day)
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        try:
            data: Dict[str, Any] = {}
            for job in self._jobs:
                if job._last_fired_date is not None:
                    data[job.name] = job._last_fired_date.isoformat()
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log.warning("Failed to save scheduler state: %s", e)

    def _load_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            for job in self._jobs:
                if job.name in data:
                    job._last_fired_date = datetime.date.fromisoformat(data[job.name])
        except Exception as e:
            log.warning("Failed to load scheduler state: %s", e)

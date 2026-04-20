"""Data refresh manager — orchestrates scheduled data updates.

Manages freshness tracking and scheduled refresh of:
- Weather data (frequent: every 30 min)
- Advisories/incidents (frequent: every 15 min via simulator)
- RAG vector index (daily)
- Base routing data (weekly/manual)
- Model retraining (weekly/manual)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RefreshStatus:
    """Status of a single data source refresh."""
    source: str
    last_refresh: datetime | None = None
    next_refresh: datetime | None = None
    status: str = "pending"  # pending, running, success, error
    error_message: str = ""
    refresh_count: int = 0
    cadence: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "next_refresh": self.next_refresh.isoformat() if self.next_refresh else None,
            "status": self.status,
            "error_message": self.error_message,
            "refresh_count": self.refresh_count,
            "cadence": self.cadence,
            "freshness_seconds": (
                (datetime.now() - self.last_refresh).total_seconds()
                if self.last_refresh else None
            ),
        }


@dataclass
class RefreshJob:
    """A scheduled refresh job."""
    name: str
    interval_seconds: float
    callback: callable
    status: RefreshStatus = field(default_factory=lambda: RefreshStatus(source="unknown"))
    enabled: bool = True


class RefreshManager:
    """Manages scheduled data refresh jobs.

    Runs background threads for periodic data updates.
    Tracks freshness metadata for each data source.
    """

    def __init__(self):
        self.jobs: dict[str, RefreshJob] = {}
        self.statuses: dict[str, RefreshStatus] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def register_job(
        self,
        name: str,
        callback: callable,
        interval_seconds: float,
        cadence_label: str = "periodic",
    ):
        """Register a periodic refresh job."""
        status = RefreshStatus(source=name, cadence=cadence_label)
        job = RefreshJob(
            name=name,
            interval_seconds=interval_seconds,
            callback=callback,
            status=status,
        )
        with self._lock:
            self.jobs[name] = job
            self.statuses[name] = status
        logger.info(f"Registered refresh job: {name} (every {interval_seconds}s)")

    def start(self):
        """Start all registered refresh jobs as background threads."""
        for name, job in self.jobs.items():
            if not job.enabled:
                continue
            thread = threading.Thread(
                target=self._run_job_loop,
                args=(job,),
                daemon=True,
                name=f"refresh-{name}",
            )
            self._threads[name] = thread
            thread.start()
            logger.info(f"Started refresh job: {name}")

    def stop(self):
        """Stop all refresh jobs."""
        self._stop_event.set()
        for name, thread in self._threads.items():
            thread.join(timeout=5)
            logger.info(f"Stopped refresh job: {name}")

    def _run_job_loop(self, job: RefreshJob):
        """Run a single job in a loop."""
        # Run immediately on start
        self._execute_job(job)

        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=job.interval_seconds):
                break
            self._execute_job(job)

    def _execute_job(self, job: RefreshJob):
        """Execute a single job and update its status."""
        status = job.status
        status.status = "running"
        try:
            job.callback()
            status.status = "success"
            status.error_message = ""
            status.refresh_count += 1
        except Exception as e:
            status.status = "error"
            status.error_message = str(e)
            logger.error(f"Refresh job {job.name} failed: {e}")
        finally:
            now = datetime.now()
            status.last_refresh = now
            status.next_refresh = now + timedelta(seconds=job.interval_seconds)

    def get_freshness(self) -> dict:
        """Get freshness status for all data sources."""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "sources": {
                    name: status.to_dict()
                    for name, status in self.statuses.items()
                },
            }

    def trigger_refresh(self, name: str) -> bool:
        """Manually trigger a specific refresh job."""
        job = self.jobs.get(name)
        if not job:
            return False
        self._execute_job(job)
        return True

"""Background agent that auto-deletes the oldest MinIO objects when
whole-system disk usage exceeds a configured threshold.

Design:
- Polls ``psutil.disk_usage("/").percent`` every ``check_interval_seconds``
  (same call used by processes/device/telemetry.py for the health payload).
- When disk used-% >= ``threshold_percent`` (e.g. 90% of a 256 GB SSD),
  it lists all objects in the bucket, sorts them oldest-first by
  ``last_modified``, then deletes everything older than
  ``delete_window_amount * delete_window_unit`` (e.g. 1 hour).
- The deletion loop repeats within a single check cycle until disk usage
  drops below the threshold or no more eligible objects remain.
- All errors on individual object removals are caught and logged — the agent
  never crashes the slave process.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

try:
    import psutil
except ImportError:  # pragma: no cover - already guarded in telemetry.py
    psutil = None  # type: ignore[assignment]

from slave.config import StorageCleanupConfig
from slave.logging.logger import get_logger
from slave.processes.minio.minio_client import MinioClientWrapper

if TYPE_CHECKING:
    pass  # Minio type used only via MinioClientWrapper.client at runtime

log = get_logger(__name__)

_VALID_UNITS = {"hours", "minutes"}


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions — easy to unit-test independently)
# ---------------------------------------------------------------------------

def _system_disk_percent(path: str = "/") -> float:
    """Return used-% of the filesystem that contains ``path``.

    Identical to the psutil call in processes/device/telemetry.py:
        psutil.disk_usage("/").percent
    Reflects the real remaining space on the physical drive (e.g. 256 GB SSD),
    not an estimate based on object sizes. Returns 0.0 if psutil is missing.
    """
    if psutil is None:
        log.warning("psutil not installed — StorageCleanupAgent cannot read disk usage")
        return 0.0
    return psutil.disk_usage(path).percent


def _list_objects_oldest_first(client, bucket: str) -> list:
    """Return all objects in ``bucket`` sorted ascending by last_modified
    (oldest first). The caller is responsible for catching SDK exceptions.
    """
    objects = list(client.list_objects(bucket, recursive=True))
    objects.sort(key=lambda o: o.last_modified)
    return objects


def _cutoff_datetime(amount: int, unit: str) -> datetime:
    """Return the UTC datetime boundary before which objects are considered
    old enough to be deleted.

    Objects with ``last_modified < _cutoff_datetime(...)`` are deletion
    candidates in the current pass.

    Args:
        amount: Number of time units.
        unit:   ``"hours"`` or ``"minutes"``.
    """
    if unit not in _VALID_UNITS:
        log.warning(
            "Unknown delete_window unit '%s' — defaulting to 'hours'. Valid: %s",
            unit,
            _VALID_UNITS,
        )
        unit = "hours"
    delta = timedelta(hours=amount) if unit == "hours" else timedelta(minutes=amount)
    return datetime.now(timezone.utc) - delta


# ---------------------------------------------------------------------------
# StorageCleanupAgent
# ---------------------------------------------------------------------------

class StorageCleanupAgent:
    """Background daemon that keeps MinIO storage in check by deleting the
    oldest objects whenever the system disk crosses the configured threshold.

    Lifecycle::

        agent = StorageCleanupAgent(minio_client, config.storage_cleanup)
        agent.start()   # spawns daemon thread — no-op if disabled
        ...
        agent.stop()    # signals thread to exit and joins with 10s timeout
    """

    def __init__(
        self,
        minio_client: MinioClientWrapper,
        config: StorageCleanupConfig,
    ) -> None:
        self._minio = minio_client
        self._config = config
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="storage-cleanup-agent",
            daemon=True,  # exits automatically when the main process dies
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread.

        Skipped silently if either ``storage_cleanup.enabled`` is false or
        MinIO itself is disabled — no objects to manage in either case.
        """
        if not self._config.enabled:
            log.info("StorageCleanupAgent disabled (storage_cleanup.enabled=false) — skipping")
            return
        if not self._minio.enabled:
            log.info(
                "StorageCleanupAgent skipped — MinIO is disabled (minio.enabled=false); "
                "nothing to clean up"
            )
            return
        log.info(
            "StorageCleanupAgent starting: threshold=%.0f%%, window=%d %s, interval=%ds",
            self._config.threshold_percent,
            self._config.delete_window_amount,
            self._config.delete_window_unit,
            self._config.check_interval_seconds,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to stop and wait up to 10 seconds."""
        self._stop.set()
        self._thread.join(timeout=10)
        log.info("StorageCleanupAgent stopped")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main polling loop — wakes every check_interval_seconds."""
        # Use Event.wait as a cancellable sleep: returns True when stop is set.
        while not self._stop.wait(timeout=self._config.check_interval_seconds):
            try:
                self._check_and_cleanup()
            except Exception:
                log.exception(
                    "StorageCleanupAgent: unhandled error during cleanup pass — will retry"
                )

    def _check_and_cleanup(self) -> None:
        """Single check-and-cleanup cycle.

        1. Read current system disk usage.
        2. If below threshold — return immediately (common path).
        3. Otherwise enter the deletion loop:
           a. List all bucket objects, sorted oldest-first.
           b. Delete everything older than ``delete_window``.
           c. Re-read disk usage (reflects actually freed space).
           d. Repeat until below threshold or no eligible objects remain.
        """
        client = self._minio.client
        bucket = self._minio.bucket

        # -- 1. Read real system disk % (same as telemetry.py payload["disk"])
        disk_pct = _system_disk_percent("/")
        log.debug(
            "StorageCleanupAgent check: disk=%.1f%% threshold=%.0f%%",
            disk_pct,
            self._config.threshold_percent,
        )

        # -- 2. Under threshold — nothing to do
        if disk_pct < self._config.threshold_percent:
            return

        log.warning(
            "System disk at %.1f%% — exceeds %.0f%% threshold. "
            "Starting MinIO cleanup pass (window: %d %s, oldest-first).",
            disk_pct,
            self._config.threshold_percent,
            self._config.delete_window_amount,
            self._config.delete_window_unit,
        )

        # -- 3. Deletion loop
        while disk_pct >= self._config.threshold_percent:
            deleted_count, freed_bytes = self._delete_one_window(client, bucket)
            if deleted_count == 0:
                log.warning(
                    "No more objects eligible for deletion but disk is still at %.1f%%. "
                    "MinIO bucket may be empty or all objects are newer than the window.",
                    disk_pct,
                )
                break
            # Re-read after deletion — reflects actual freed space, not arithmetic estimate
            disk_pct = _system_disk_percent("/")
            log.info(
                "Cleanup pass complete: deleted %d objects, freed ~%.2f MB. "
                "Disk now at %.1f%%.",
                deleted_count,
                freed_bytes / 1e6,
                disk_pct,
            )

    def _delete_one_window(self, client, bucket: str) -> tuple[int, int]:
        """Delete all objects older than the configured time window.

        Returns:
            (deleted_count, freed_bytes) — both 0 if nothing was eligible.
        """
        try:
            objects = _list_objects_oldest_first(client, bucket)
        except Exception:
            log.exception("StorageCleanupAgent: failed to list bucket objects")
            return 0, 0

        if not objects:
            return 0, 0

        cutoff = _cutoff_datetime(
            self._config.delete_window_amount,
            self._config.delete_window_unit,
        )
        candidates = [o for o in objects if o.last_modified < cutoff]

        if not candidates:
            log.debug(
                "No objects older than cutoff %s in bucket '%s'",
                cutoff.isoformat(),
                bucket,
            )
            return 0, 0

        log.info(
            "Deleting %d objects older than %s (cutoff: %s)",
            len(candidates),
            f"{self._config.delete_window_amount} {self._config.delete_window_unit}",
            cutoff.isoformat(),
        )

        deleted, freed = 0, 0
        for obj in candidates:
            if self._stop.is_set():
                log.info("StorageCleanupAgent: stop requested mid-pass — aborting deletion loop")
                break
            try:
                client.remove_object(bucket, obj.object_name)
                freed += obj.size or 0
                deleted += 1
                log.debug(
                    "Deleted object: %s (last_modified=%s, size=%d bytes)",
                    obj.object_name,
                    obj.last_modified.isoformat() if obj.last_modified else "unknown",
                    obj.size or 0,
                )
            except Exception:
                log.exception(
                    "Failed to delete object '%s' — skipping and continuing",
                    obj.object_name,
                )

        return deleted, freed

"""Thread-safe lifecycle state for background compute jobs.

The Dash UI polls long-running RF, MDS, Pseudo-ESS, and consensus-tree
computations.  A completed job must not become a one-shot event: an HTTP
response can be superseded before the browser applies it.  ``JobManager``
therefore keeps terminal state until the matching browser generation
acknowledges it.

This module deliberately has no Dash or worker imports.  Job-specific code
submits work with a success finalizer that performs its domain side effects
once and returns a small, JSON-friendly terminal payload.  Polling reads
snapshots and never consumes them.
"""

from __future__ import annotations

import copy
import math
import time
import uuid
from concurrent.futures import CancelledError, Executor, Future
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any, Callable, Mapping


class JobState(StrEnum):
    """Lifecycle states exposed to polling and diagnostics."""

    QUEUED = "queued"
    RUNNING = "running"
    FINALIZING = "finalizing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset(
    {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}
)


class JobBusyError(RuntimeError):
    """Raised when a single-active manager already owns a live job."""

    def __init__(self, active: "JobRef") -> None:
        self.active = active
        super().__init__(
            f"background job {active.job_id} ({active.kind}) is still active"
        )


@dataclass(frozen=True, slots=True)
class JobRef:
    """Immutable identity passed between the server and a browser store."""

    job_id: str
    generation: int
    kind: str
    owner_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "generation": self.generation,
            "kind": self.kind,
            "owner_id": self.owner_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JobRef":
        return cls(
            job_id=str(value["job_id"]),
            generation=int(value["generation"]),
            kind=str(value["kind"]),
            owner_id=(
                None
                if value.get("owner_id") is None
                else str(value["owner_id"])
            ),
        )


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """The latest valid progress observation for a job."""

    fraction: float
    phase: str
    label: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "phase": self.phase,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    """A stable terminal event retained until browser acknowledgement."""

    state: JobState
    revision: int
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "revision": self.revision,
            "payload": copy.deepcopy(dict(self.payload)),
        }


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Immutable point-in-time view returned to poll callbacks."""

    ref: JobRef
    state: JobState
    progress: ProgressSnapshot | None
    terminal: TerminalEvent | None
    metadata: Mapping[str, Any]
    acknowledged: bool
    delivery_attempt: int
    submitted_at: float
    started_at: float | None
    finished_at: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return a small structure suitable for a ``dcc.Store``."""

        value = self.ref.as_dict()
        value.update(
            {
                "state": self.state.value,
                "progress": (
                    None if self.progress is None else self.progress.as_dict()
                ),
                "terminal": (
                    None if self.terminal is None else self.terminal.as_dict()
                ),
                "metadata": copy.deepcopy(dict(self.metadata)),
                "acknowledged": self.acknowledged,
                "delivery_attempt": self.delivery_attempt,
                "submitted_at": self.submitted_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }
        )
        return value


Finalizer = Callable[[JobRef, Any], Mapping[str, Any] | None]


@dataclass(slots=True)
class _JobRecord:
    ref: JobRef
    state: JobState
    metadata: dict[str, Any]
    submitted_at: float
    future: Future[Any] | None = None
    progress: ProgressSnapshot | None = None
    terminal: TerminalEvent | None = None
    acknowledged: bool = False
    delivery_attempt: int = 0
    terminal_revision: int = 0
    started_at: float | None = None
    finished_at: float | None = None


class JobManager:
    """Own background-job state and make terminal delivery replayable.

    By default only one unacknowledged job is accepted at a time.  This
    mirrors TreeTracer's single-thread executor and serialized persistent
    worker instead of silently building a queue behind independently enabled
    compute buttons.

    The manager does not own the supplied executor.  Callers remain
    responsible for executor shutdown and for interrupting opaque native work
    when a running job is cancelled.
    """

    def __init__(
        self,
        *,
        single_active: bool = True,
        max_history: int = 32,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if max_history < 1:
            raise ValueError("max_history must be at least 1")
        self._single_active = single_active
        self._max_history = max_history
        self._clock = clock
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._lock = RLock()
        self._records: dict[str, _JobRecord] = {}
        self._next_generation = 1

    def submit(
        self,
        executor: Executor,
        kind: str,
        fn: Callable[..., Any],
        /,
        *args: Any,
        owner_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        finalizer: Finalizer | None = None,
        cancel_exceptions: tuple[type[BaseException], ...] = (),
        **kwargs: Any,
    ) -> JobRef:
        """Submit work and attach exactly one terminal-state handler.

        ``finalizer`` runs once after successful compute completion.  It may
        persist the raw result and must return only the small payload needed by
        the UI.  A finalizer exception becomes a sticky failed terminal event.

        Exceptions raised by ``executor.submit`` are also captured as a failed
        terminal event, so a browser that already received the returned job
        identity can render the error through the normal polling path.
        """

        kind = str(kind).strip()
        if not kind:
            raise ValueError("kind must be a non-empty string")
        if not isinstance(cancel_exceptions, tuple) or not all(
            isinstance(exc_type, type)
            and issubclass(exc_type, BaseException)
            for exc_type in cancel_exceptions
        ):
            raise TypeError("cancel_exceptions must be a tuple of exception types")

        with self._lock:
            active = self._active_ref_locked()
            if self._single_active and active is not None:
                raise JobBusyError(active)

            generation = self._next_generation
            self._next_generation += 1
            ref = JobRef(
                job_id=self._id_factory(),
                generation=generation,
                kind=kind,
                owner_id=owner_id,
            )
            if ref.job_id in self._records:
                raise ValueError(f"duplicate job id from id_factory: {ref.job_id}")
            record = _JobRecord(
                ref=ref,
                state=JobState.QUEUED,
                metadata=copy.deepcopy(dict(metadata or {})),
                submitted_at=self._clock(),
            )
            self._records[ref.job_id] = record
            self._prune_history_locked()

        def run() -> None:
            if not self._mark_running(ref):
                return
            self._run_job(
                ref,
                fn,
                args,
                kwargs,
                finalizer=finalizer,
                cancel_exceptions=cancel_exceptions,
            )

        try:
            future = executor.submit(run)
        except BaseException as exc:
            self._finish_exception(ref, exc, stage="submit")
            return ref

        with self._lock:
            current = self._matching_record_locked(ref)
            if current is not None:
                current.future = future

        # This callback must remain tiny.  ``Future.add_done_callback`` invokes
        # it synchronously when a very fast future finished before registration;
        # running domain finalization here would then block ``submit`` itself.
        future.add_done_callback(
            lambda completed: self._release_future(ref, completed)
        )
        return ref

    def snapshot(self, ref: JobRef) -> JobSnapshot | None:
        """Read state without consuming or changing terminal delivery."""

        with self._lock:
            record = self._matching_record_locked(ref)
            return None if record is None else self._snapshot_locked(record)

    def snapshot_for_delivery(self, ref: JobRef) -> JobSnapshot | None:
        """Read state for a poll response and count terminal replays.

        A changing delivery attempt lets a ``dcc.Store`` retrigger browser
        reconciliation even though the semantic terminal event and its
        revision remain stable.
        """

        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None:
                return None
            if record.terminal is not None and not record.acknowledged:
                record.delivery_attempt += 1
            return self._snapshot_locked(record)

    def active_ref(self) -> JobRef | None:
        """Return the current non-acknowledged job, if any."""

        with self._lock:
            return self._active_ref_locked()

    def update_progress(
        self,
        ref: JobRef,
        fraction: float,
        phase: str,
        label: str | None = None,
    ) -> bool:
        """Publish monotonic progress for a matching live job.

        Stale generations and jobs already finalizing/terminal return ``False``.
        Fractions are clamped to ``[0, 1]`` and never move backwards.
        """

        fraction = float(fraction)
        if not math.isfinite(fraction):
            raise ValueError("progress fraction must be finite")
        fraction = max(0.0, min(fraction, 1.0))
        phase = str(phase).strip() or "running"

        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None or record.state not in {
                JobState.QUEUED,
                JobState.RUNNING,
            }:
                return False
            if record.progress is not None:
                fraction = max(record.progress.fraction, fraction)
            record.progress = ProgressSnapshot(fraction, phase, label)
            return True

    def acknowledge(
        self,
        ref: JobRef,
        terminal_revision: int,
    ) -> bool:
        """Acknowledge only the exact terminal event applied by the browser."""

        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None or record.terminal is None:
                return False
            if record.terminal.revision != int(terminal_revision):
                return False
            record.acknowledged = True
            self._prune_history_locked()
            return True

    def mark_cancelled(
        self,
        ref: JobRef,
        *,
        message: str = "Background computation cancelled",
    ) -> bool:
        """Publish a sticky cancellation for queued/running work.

        The caller must separately interrupt work that ``Future.cancel()``
        cannot stop (for TreeTracer, this means killing the persistent worker).
        Cancellation loses to finalization once irreversible domain side
        effects have begun.
        """

        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None or record.state not in {
                JobState.QUEUED,
                JobState.RUNNING,
            }:
                return False
            future = record.future
            self._set_terminal_locked(
                record,
                JobState.CANCELLED,
                {"message": str(message), "stage": "compute"},
            )

        if future is not None:
            future.cancel()
        return True

    def forget(self, ref: JobRef) -> bool:
        """Remove an acknowledged terminal record from retained history."""

        with self._lock:
            record = self._matching_record_locked(ref)
            if (
                record is None
                or record.terminal is None
                or not record.acknowledged
            ):
                return False
            del self._records[ref.job_id]
            return True

    def invalidate(self, ref: JobRef) -> bool:
        """Forget a job immediately and reject any late completion.

        This is the reset/clear-data operation, not normal browser delivery.
        Removing the record makes every later state transition from the job's
        wrapper fail its identity lookup, so a stale result cannot be finalized
        into freshly cleared application state.  Running native work still has
        to be interrupted separately by its owner.
        """

        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None:
                return False
            future = record.future
            del self._records[ref.job_id]

        if future is not None:
            future.cancel()
        return True

    def _mark_running(self, ref: JobRef) -> bool:
        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None or record.state is not JobState.QUEUED:
                return False
            record.state = JobState.RUNNING
            record.started_at = self._clock()
            return True

    def _run_job(
        self,
        ref: JobRef,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        finalizer: Finalizer | None,
        cancel_exceptions: tuple[type[BaseException], ...],
    ) -> None:
        try:
            result = fn(*args, **kwargs)
        except CancelledError as exc:
            if self._claim_finalization(ref):
                self._finish_cancelled(ref, exc)
            return
        except BaseException as exc:
            if self._claim_finalization(ref):
                if cancel_exceptions and isinstance(exc, cancel_exceptions):
                    self._finish_cancelled(ref, exc)
                else:
                    self._finish_exception(ref, exc, stage="compute")
            return

        # Explicit cancellation can win while the opaque compute function is
        # running.  In that case its late result is intentionally discarded.
        if not self._claim_finalization(ref):
            return

        try:
            payload = {} if finalizer is None else finalizer(ref, result)
            if payload is None:
                payload = {}
            if not isinstance(payload, Mapping):
                raise TypeError("job finalizer must return a mapping or None")
            payload = copy.deepcopy(dict(payload))
        except BaseException as exc:
            self._finish_exception(ref, exc, stage="finalize")
            return

        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None or record.state is not JobState.FINALIZING:
                return
            self._set_terminal_locked(record, JobState.SUCCEEDED, payload)

    def _claim_finalization(self, ref: JobRef) -> bool:
        """Atomically grant one caller permission to finalize a job."""

        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None or record.state not in {
                JobState.QUEUED,
                JobState.RUNNING,
            }:
                return False
            record.state = JobState.FINALIZING
            return True

    def _release_future(self, ref: JobRef, future: Future[Any]) -> None:
        """Drop the completed Future without doing domain work.

        The wrapped job normally records its own terminal state.  The only
        additional case handled here is an externally cancelled queued future,
        whose wrapper never had a chance to run.
        """

        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None:
                return
            if record.future is future:
                record.future = None
            if future.cancelled() and record.state is JobState.QUEUED:
                self._set_terminal_locked(
                    record,
                    JobState.CANCELLED,
                    {
                        "message": "Background computation cancelled",
                        "error_type": "CancelledError",
                        "stage": "compute",
                    },
                )

    def _finish_cancelled(self, ref: JobRef, exc: BaseException) -> None:
        message = str(exc) or "Background computation cancelled"
        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None or record.state is not JobState.FINALIZING:
                return
            self._set_terminal_locked(
                record,
                JobState.CANCELLED,
                {
                    "message": message,
                    "error_type": type(exc).__name__,
                    "stage": "compute",
                },
            )

    def _finish_exception(
        self,
        ref: JobRef,
        exc: BaseException,
        *,
        stage: str,
    ) -> None:
        with self._lock:
            record = self._matching_record_locked(ref)
            if record is None:
                return
            if stage != "submit" and record.state is not JobState.FINALIZING:
                return
            if stage == "submit" and record.state not in {
                JobState.QUEUED,
                JobState.RUNNING,
            }:
                return
            self._set_terminal_locked(
                record,
                JobState.FAILED,
                {
                    "message": str(exc) or type(exc).__name__,
                    "error_type": type(exc).__name__,
                    "stage": stage,
                },
            )

    def _set_terminal_locked(
        self,
        record: _JobRecord,
        state: JobState,
        payload: Mapping[str, Any],
    ) -> None:
        if state not in TERMINAL_STATES:
            raise ValueError(f"{state!r} is not a terminal state")
        record.state = state
        record.future = None
        record.finished_at = self._clock()
        record.terminal_revision += 1
        record.terminal = TerminalEvent(
            state=state,
            revision=record.terminal_revision,
            payload=copy.deepcopy(dict(payload)),
        )
        if state is JobState.SUCCEEDED:
            record.progress = ProgressSnapshot(1.0, "done", "complete")
        else:
            last_fraction = (
                0.0 if record.progress is None else record.progress.fraction
            )
            record.progress = ProgressSnapshot(last_fraction, state.value)

    def _matching_record_locked(self, ref: JobRef) -> _JobRecord | None:
        record = self._records.get(ref.job_id)
        if record is None or record.ref != ref:
            return None
        return record

    def _active_ref_locked(self) -> JobRef | None:
        for record in reversed(tuple(self._records.values())):
            if not record.acknowledged:
                return record.ref
        return None

    def _snapshot_locked(self, record: _JobRecord) -> JobSnapshot:
        progress = record.progress
        terminal = record.terminal
        return JobSnapshot(
            ref=record.ref,
            state=record.state,
            progress=(
                None
                if progress is None
                else ProgressSnapshot(
                    progress.fraction,
                    progress.phase,
                    progress.label,
                )
            ),
            terminal=(
                None
                if terminal is None
                else TerminalEvent(
                    terminal.state,
                    terminal.revision,
                    copy.deepcopy(dict(terminal.payload)),
                )
            ),
            metadata=copy.deepcopy(record.metadata),
            acknowledged=record.acknowledged,
            delivery_attempt=record.delivery_attempt,
            submitted_at=record.submitted_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )

    def _prune_history_locked(self) -> None:
        excess = len(self._records) - self._max_history
        if excess <= 0:
            return
        removable = [
            job_id
            for job_id, record in self._records.items()
            if record.acknowledged
        ]
        for job_id in removable[:excess]:
            del self._records[job_id]


# Process-local coordinator used by the desktop app.  Browser-mode migration
# must provide ``owner_id`` values before supporting multiple independent tabs.
job_manager = JobManager()

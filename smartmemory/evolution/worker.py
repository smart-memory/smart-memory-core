"""CORE-EVO-LIVE-1: Evolution worker daemon thread.

Consumes MutationEvents from the queue, debounces, coalesces, routes to
incremental evolvers, and flushes writes through the shared backend.
Runs idle catch-up (full batch evolution) after 60s of silence.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional

from smartmemory.evolution.batcher import WriteBatcher
from smartmemory.evolution.events import EvolutionAction, EvolutionContext, MutationEvent
from smartmemory.evolution.queue import EvolutionQueue
from smartmemory.evolution.router import EvolutionRouter

if TYPE_CHECKING:
    from smartmemory.graph.backends.backend import SmartGraphBackend
    from smartmemory.graph.compute import GraphComputeLayer

logger = logging.getLogger(__name__)

# Tuning constants
DEBOUNCE_SECONDS = 0.05  # 50ms debounce window
IDLE_TIMEOUT_SECONDS = 60.0  # Run full batch cycle after 60s silence


class EvolutionWorker(threading.Thread):
    """Daemon thread that processes mutation events into incremental evolution.

    Lifecycle:
    - Lazy-started on first EvolutionQueue.put() (via ensure_started()).
    - Main loop: wait(50ms) → drain → coalesce → route → evolve_incremental → flush.
    - Idle: after 60s with no events, runs full batch evolution cycle.
    - Shutdown: set stop event → drain remaining → flush → exit.
    """

    def __init__(
        self,
        queue: EvolutionQueue,
        router: EvolutionRouter,
        backend: SmartGraphBackend,
        graph: GraphComputeLayer,
        memory: Any = None,
    ) -> None:
        super().__init__(daemon=True, name="evolution-worker")
        self._queue = queue
        self._router = router
        self._backend = backend
        self._graph = graph
        self._memory = memory  # For idle catch-up and run_batch_evolver
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._skip_drain = False  # Set by shutdown(drain=False)
        self._last_activity = time.monotonic()
        self._idle_catchup_running = False

    @property
    def is_active(self) -> bool:
        """True if the worker thread is alive and not stopped."""
        return self.is_alive() and not self._stop_event.is_set()

    def ensure_started(self) -> None:
        """Start the worker thread if not already running."""
        if not self.is_alive() and not self._stop_event.is_set():
            try:
                self.start()
                self._started_event.wait(timeout=2.0)
            except RuntimeError:
                pass  # Already started

    def run(self) -> None:
        """Main worker loop."""
        logger.info("EvolutionWorker: started")
        self._started_event.set()

        while not self._stop_event.is_set():
            # Wait for events or timeout (for idle detection)
            signalled = self._queue.wait(timeout=DEBOUNCE_SECONDS)

            if self._stop_event.is_set():
                break

            # Drain and process
            events = self._queue.drain()
            if events:
                self._last_activity = time.monotonic()
                self._process_events(events)
            elif not signalled:
                # No events — check for idle catch-up
                idle_time = time.monotonic() - self._last_activity
                if idle_time >= IDLE_TIMEOUT_SECONDS and not self._idle_catchup_running:
                    self._run_idle_catchup()
                    self._last_activity = time.monotonic()

        # Drain any remaining events on shutdown (only if drain was requested)
        if not self._skip_drain:
            remaining = self._queue.drain()
            if remaining:
                self._process_events(remaining)

        logger.info("EvolutionWorker: stopped")

    def _process_events(self, events: List[MutationEvent]) -> None:
        """Coalesce, route, evolve, and flush a batch of events."""
        coalesced = EvolutionQueue.coalesce(events)
        batcher = WriteBatcher()

        for event in coalesced:
            evolvers = self._router.route(event)
            for evolver in evolvers:
                ctx = EvolutionContext(
                    event=event,
                    graph=self._graph,
                    backend=self._backend,
                    workspace_id=event.workspace_id,
                )
                try:
                    actions: List[EvolutionAction] = evolver.evolve_incremental(ctx)
                    batcher.add_all(actions)
                except Exception as exc:
                    logger.warning(
                        "EvolutionWorker: evolver %s failed on %s: %s",
                        type(evolver).__name__,
                        event.item_id,
                        exc,
                    )

        if batcher.pending_count > 0:
            try:
                count = batcher.flush(self._backend, self._memory)
                logger.debug("EvolutionWorker: flushed %d actions", count)
            except Exception as exc:
                logger.error("EvolutionWorker: flush failed: %s", exc)

    def _run_idle_catchup(self) -> None:
        """Run a full batch evolution cycle as consistency catch-up."""
        if not self._memory:
            return

        self._idle_catchup_running = True
        try:
            from smartmemory.evolution.cycle import run_evolution_cycle

            logger.debug("EvolutionWorker: running idle catch-up (full batch cycle)")
            run_evolution_cycle(self._memory, logger=logger)
        except Exception as exc:
            logger.warning("EvolutionWorker: idle catch-up failed: %s", exc)
        finally:
            self._idle_catchup_running = False

    def shutdown(self, drain: bool = True, timeout: float = 5.0) -> None:
        """Stop the worker thread.

        Args:
            drain: If True, process remaining events before stopping.
            timeout: Max seconds to wait for the thread to finish.
        """
        logger.info("EvolutionWorker: shutting down (drain=%s)", drain)
        if not drain:
            self._skip_drain = True
        self._stop_event.set()
        self._queue._event.set()  # Wake up if blocked on wait()

        if self.is_alive():
            self.join(timeout=timeout)
            if self.is_alive():
                logger.warning("EvolutionWorker: thread did not stop within %.1fs", timeout)

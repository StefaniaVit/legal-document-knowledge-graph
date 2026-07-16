"""
Wraps entity_relation_extractor.extract() with a hard per-call timeout, by
running it in a persistent worker subprocess that gets killed and replaced
if a call runs too long.

This exists because of an observed, not-fully-understood failure mode with
Qwen2.5-7B: an occasional call (~2.5% of chunks in testing) takes 5-10
minutes instead of the usual 5-20 seconds. One trigger was identified and
fixed (a few-shot example the model would sometimes regurgitate on
thematically similar text), but a residual, rarer occurrence of the same
symptom remained with no further identified cause. Rather than keep
debugging an intermittent, hard-to-reproduce slowdown, this bounds the
worst-case damage: any single call that exceeds the timeout gets the whole
worker process killed (llama.cpp inference can't be cancelled any other
way -- there's no API to abort a call already in progress), retried once
against a fresh worker, and skipped if it times out again.

The worker is a genuine OS subprocess, not a thread: a stuck llama.cpp call
runs in a C extension that doesn't release control back to Python in a way
a thread-based timeout could interrupt, but a whole process can be sent
SIGTERM/SIGKILL regardless of what it's doing internally.
"""
import multiprocessing as mp
import queue as queue_module
from typing import Optional

DEFAULT_TIMEOUT_SECONDS = 60
WORKER_STARTUP_TIMEOUT_SECONDS = 60


def _worker_loop(task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Runs in the subprocess. Loads the model once, then services one
    extract() call at a time for as long as the process lives."""
    from src.extraction.entity_relation_extractor import extract

    result_queue.put(("ready", None))
    while True:
        chunk_text = task_queue.get()
        if chunk_text is None:  # sentinel: shut down cleanly
            return
        try:
            result_queue.put(("ok", extract(chunk_text)))
        except Exception as e:
            result_queue.put(("error", str(e)))


class TimeoutSafeExtractor:
    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.timeout = timeout
        self._ctx = mp.get_context("spawn")
        self.task_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.process: Optional[mp.Process] = None
        self._start_worker()

    def _start_worker(self) -> None:
        self.task_queue = self._ctx.Queue()
        self.result_queue = self._ctx.Queue()
        self.process = self._ctx.Process(
            target=_worker_loop, args=(self.task_queue, self.result_queue), daemon=True,
        )
        self.process.start()
        # Block until the model has actually finished loading in the worker,
        # so the first real extract() call isn't the one paying that cost
        # (and so a slow *load* isn't mistaken for a slow *call* later).
        status, _ = self.result_queue.get(timeout=WORKER_STARTUP_TIMEOUT_SECONDS)
        assert status == "ready"

    def _kill_worker(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)

    def extract(self, chunk_text: str) -> dict:
        """Returns the extract() result. Raises TimeoutError if the call
        exceeds the timeout twice in a row (once retried against a fresh
        worker), or RuntimeError if the worker itself raised."""
        for attempt in range(2):
            self.task_queue.put(chunk_text)
            try:
                status, payload = self.result_queue.get(timeout=self.timeout)
            except queue_module.Empty:
                self._kill_worker()
                self._start_worker()
                if attempt == 1:
                    raise TimeoutError(f"extraction call exceeded {self.timeout}s twice in a row")
                continue
            if status == "error":
                raise RuntimeError(payload)
            return payload
        raise AssertionError("unreachable")

    def close(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.task_queue.put(None)
            self.process.join(timeout=5)
            self._kill_worker()

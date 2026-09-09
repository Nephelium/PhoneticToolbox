"""Owned P07 expiry loop; no directory-tree removal or generic process cleanup."""
import time
import logging


def run_cleanup(storage, stop):
    warned = False
    while not stop.is_set():
        try:
            status = storage.cleanup()
            warned = False
            due = status['next_due']
            # Wake early without deleting early. Within the final minute check at
            # most every second; every deletion still compares database UTC time.
            delay = min(30, max(0.1, due-time.time())) if due is not None else 30
            if due is not None and due-time.time() <= 60:
                delay = min(delay, 1)
            stop.wait(delay)
        except Exception:
            # A failure never resets accounting. Runtime diagnostics contain no paths.
            if not warned:
                logging.getLogger(__name__).warning('storage_cleanup_retry_required')
                warned = True
            stop.wait(1)

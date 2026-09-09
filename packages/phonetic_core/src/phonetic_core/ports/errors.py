"""Lightweight errors shared by computation and I/O adapters."""


class BackendAborted(Exception):
    """Host cancellation/resource failure must abort, never become a missing track."""

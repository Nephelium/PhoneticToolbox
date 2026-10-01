"""Lightweight errors shared by computation and I/O adapters."""


class BackendAborted(Exception):
    """Host cancellation/resource failure must abort, never become a missing track."""


ACOUSTIC_STAGES = frozenset({'energy', 'formants', 'praat_pitch', 'reaper',
    'spectrum', 'tilt', 'correction', 'cpp', 'hnr', 'shr', 'slope', 'soe', 'jitter_shimmer'})


class AcousticComputationError(RuntimeError):
    """A fixed stage code; private exception text never crosses the process boundary."""
    def __init__(self, stage):
        if stage not in ACOUSTIC_STAGES:
            raise ValueError('Unknown acoustic stage')
        self.stage = stage
        self.code = f'analysis_{stage}_failed'
        super().__init__(self.code)


def raise_stage_failure(stage, error):
    if isinstance(error, (BackendAborted, MemoryError, AcousticComputationError)):
        raise error
    raise AcousticComputationError(stage) from error

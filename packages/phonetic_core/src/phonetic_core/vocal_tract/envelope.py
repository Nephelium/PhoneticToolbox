"""Audition envelopes shared by live, prepared and exported sound."""
import numpy as np

ATTACK_SECONDS = .050
RELEASE_SECONDS = .015


def attack_envelope(offset, count, sample_rate):
    """Raised-cosine attack measured in samples, continuous across blocks."""
    length = max(2, round(sample_rate * ATTACK_SECONDS))
    phase = np.clip((offset + np.arange(count)) / (length - 1), 0, 1)
    return .5 - .5 * np.cos(np.pi * phase)


def sequence_envelope(count, sample_rate, silent_intervals):
    """Fade only utterance edges and explicit pauses, never adjacent poses."""
    envelope = np.zeros(count, dtype=np.float64)
    start = 0
    for pause_start, pause_end in [*silent_intervals, (count, count)]:
        end = min(count, pause_start)
        length = end - start
        if length > 0:
            envelope[start:end] = 1.
            attack = min(round(sample_rate * ATTACK_SECONDS), length // 2)
            release = min(round(sample_rate * RELEASE_SECONDS), length // 2)
            if attack > 1:
                envelope[start:start + attack] *= .5 - .5 * np.cos(np.linspace(0, np.pi, attack))
            if release > 1:
                envelope[end - release:end] *= .5 + .5 * np.cos(np.linspace(0, np.pi, release))
        start = max(start, pause_end)
    return envelope

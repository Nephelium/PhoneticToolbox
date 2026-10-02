"""M16 pure local-recording operations. No device, Qt or filesystem access.

source_id: ptb-m16-recording-20261002. Metering design reviewed against
VoiceVista egg_recorder; implementation deliberately preserves pre-gain PCM.
"""
from .signal import meter, spectrum, noise_profile, denoise, gain_audio
from .edits import frames, select, remove, insert

__all__ = ['meter', 'spectrum', 'noise_profile', 'denoise', 'gain_audio', 'frames', 'select', 'remove', 'insert']

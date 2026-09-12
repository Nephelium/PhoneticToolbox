"""Image magnitude reconstruction, migrated from PhoneticToolbox v2.

Griffin & Lim (1984), DOI 10.1109/TASSP.1984.1164317. Legacy grayscale
power mapping and linear resampling are deliberately retained. No host IO.
"""
from .reconstruction import reconstruct

__all__ = ['reconstruct']

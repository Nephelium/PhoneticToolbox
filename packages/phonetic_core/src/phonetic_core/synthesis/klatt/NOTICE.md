# M06 source notice

tdklatt: Copyright (c) 2017 Adrian Y. Cho and Daniel R Guest. MIT, see LICENSE.
Source ID SRC-TDKLATT; method REF-KLATT. The direct source is the local V2
snapshot, not a claim that the current upstream commit is the included version.

V3 removes tdklatt device playback, file-saving methods, import-time sys.exit,
and executable examples. Numerical classes retained V2 behavior through M06-R1.
The widget's ParameterCurve and array orchestration were extracted into engine.py.
Configuration validation, immutable task snapshots, parameter serialization and
scientific-process I/O are separate V3 boundaries. Original V2 files remain intact.

Parameter extraction reuses the previously migrated acoustic core (SRC-PRAAT and
its individual method notices). It retains V2 curve interpolation/clipping and
fallback-to-default rules; it does not establish perceptual or physiological validity.

Source hashes and manual discrepancies: docs/modules/evidence/M06-source-map.md.

2026-10-02 M06-R2: the user authorized direct scientific correction. M06/2
replaces source gain calibration and source-off behavior, fixes the noise FIR,
uses a 20 kHz internal rate, removes automatic amplitude normalization and the
HNR-to-AH mapping, and revises extraction initialization. source_levels.py is a
project digital calibration, not original Klatt code or physical SPL calibration.
No newly downloaded upstream implementation was incorporated. The original MIT
copyright and license remain. See docs/decisions/ADR-M06-R2.md and the R2 report.

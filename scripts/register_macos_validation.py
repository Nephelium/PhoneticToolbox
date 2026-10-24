"""Bind successful Mac runtime and numerical reports to the current source."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {'numpy': '2.2.6', 'scipy': '1.16.3', 'praat-parselmouth': '0.4.7',
            'pandas': '2.3.3', 'matplotlib': '3.10.8'}
OPERATIONS = ('acoustic_analysis', 'textgrid_segment', 'egg_analysis', 'lpc_analysis',
              'speech_synthesis', 'phonation_synthesis', 'pitch_manipulation',
              'phonology_induction', 'spectrogram_to_audio', 'lip_analysis')
CHECKS = {
    'm06_generate', 'm06_synthesize', 'm06_extract', 'm08_preview', 'm08_transform',
    'm08_linear', 'm07_analysis', 'm07_generation', 'egg_preview', 'egg_batch',
    'egg_inverse', 'lpc_plot_audio', 'm01_disabled', 'm01_native_required',
    'm01_streamed_local_files', 'm02_parameter_bundle_view',
    'egg_persistent_file_roi_swap_reaper_and_close',
    'egg_125_second_file_backed_two_distant_rois', 'textgrid_exact_sample_slices',
    'coreaudio_device_enumeration', 'm14_preview_three_exports',
    'm09_parameter_preview', 'm09_image_reconstruction',
    'vtl_six_presets_three_resolutions', 'mp4_h264_aac_decode_frame_timestamps',
    'lip_actual_facemesh_no_face_missing_values',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_reports(runtime, numeric, root):
    from macos_validation_identity import runtime_hashes,source_hashes,require_identity
    if not runtime.get('success') or not numeric.get('success'):
        raise RuntimeError('Both verification reports must pass')
    if not CHECKS.issubset(runtime.get('checks', [])):
        raise RuntimeError('Required scientific runtime coverage is incomplete')
    if runtime.get('runtime', {}).get('versions') != VERSIONS:
        raise RuntimeError('Scientific runtime versions differ from the validation policy')
    hashes=runtime_hashes(root)
    require_identity(runtime.get('source_sha256'),hashes,'Runtime report')
    require_identity(numeric.get('source_sha256'),source_hashes(root),'Numerical report')
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--numerics', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Run on the validated macOS arm64 source environment')
    runtime = json.loads(args.runtime.read_text())
    numeric = json.loads(args.numerics.read_text())
    hashes=validate_reports(runtime,numeric,ROOT)
    report = dict(schema='ptb-macos-validation/1', versions=VERSIONS,
                  operations={name: True for name in OPERATIONS}, source_sha256=hashes,
                  policy='Existing numerical thresholds retained; current-source runtime and capture identities verified.',
                  runtime_checks=runtime['checks'],
                  evidence_sha256={str(path): sha256(path) for path in (args.runtime, args.numerics)})
    target = args.output or ROOT / 'resources/macos-validation.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    pending=target.with_suffix('.json.pending')
    pending.write_text(json.dumps(report, indent=2) + '\n',encoding='utf8')
    pending.replace(target)
    print('Bound macOS validation to', len(hashes), 'source/native files:', target)


if __name__ == '__main__':
    main()

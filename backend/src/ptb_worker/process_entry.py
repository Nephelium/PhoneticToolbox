"""One bounded dispatch contract for Python and frozen worker processes."""
import runpy
import sys

MODULES = frozenset({
    'ptb_worker.acoustic_stream_child', 'ptb_worker.parameter_bundle_child',
    'ptb_worker.m06_child',
    'ptb_worker.m07_child',
    'ptb_worker.m14_child',
    'ptb_worker.m08_child', 'ptb_worker.cli', 'ptb_worker.core_child', 'ptb_worker.science_child',
    'ptb_worker.segment_child', 'ptb_worker.spec2wav_child', 'ptb_worker.spec2wav_preview',
    'ptb_worker.spectrogram_preview', 'ptb_worker.parameter_preview',
    'ptb_worker.legacy_conversion', 'ptb_worker.io.export_worker',
})


def command(module, *arguments):
    if module not in MODULES:
        raise ValueError('Unsupported worker entry')
    prefix = [sys.executable, '--ptb-worker', module] if getattr(sys, 'frozen', False) else [sys.executable, '-B', '-m', module]
    return prefix + list(arguments)


def dispatch(module, arguments):
    if module not in MODULES:
        raise ValueError('Unsupported worker entry')
    # Only fixed application modules can be executed. Preserve each __main__
    # protocol error wrapper and exit status, including named-pipe failures.
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name='__main__', alter_sys=True)

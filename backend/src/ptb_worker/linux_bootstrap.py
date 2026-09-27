"""Fixed Linux child entry, invoked only after systemd installs group limits."""
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    profile_path, entry, request = sys.argv[1:]
    # The profile is selected by the trusted host, never by a submitted task.
    profile = json.loads(Path(profile_path).read_text('utf-8'))
    sys.path[:0] = profile['sys_paths']
    os.environ['PTB_LINUX_RUNTIME_PROFILE'] = profile_path
    os.environ['MPLCONFIGDIR'] = profile['cache']
    os.environ['MPLBACKEND'] = 'Agg'
    from ptb_worker.native.linux_runtime import ENTRIES, fingerprint
    fingerprint()
    if profile.get('fonts'):
        from matplotlib import font_manager
        for font in profile['fonts']:
            if font not in profile['hashes']:
                raise ValueError('Unregistered runtime font')
            font_manager.fontManager.addfont(font)
    if entry not in ENTRIES:
        raise ValueError('Unsupported fixed Linux entry')
    # Existing file protocols write directly to stdout's kernel pipe. No disk
    # output bypasses quota. Ordinary Python prints must not corrupt the bundle.
    module = ENTRIES[entry]
    sys.argv = [module, request, '/dev/stdout']
    if entry == 'fonts':
        from ptb_worker.fonts import check_fonts
        from ptb_api.font_models import FigureFontSnapshot
        result = check_fonts(FigureFontSnapshot.model_validate_json(sys.stdin.buffer.readline(8192)))
        sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode())
    elif entry == 'spectrogram':
        from ptb_worker.spectrogram_preview import child
        child()
    else:
        sys.stdout = sys.stderr
        if entry == 'lpc':
            from ptb_worker.lpc_child import run
            run()
        else:
            runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()

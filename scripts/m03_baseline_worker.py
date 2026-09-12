"""M03-A test-only reader of original v2. Never import from a product entry point."""
import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import shutil
import sys
import threading
import warnings
from pathlib import Path

from baseline_support import sha, write_json


def main():
    request = json.loads(Path(sys.argv[1]).read_text('utf-8'))
    source = Path(request['source_root']).resolve()
    output = Path(request['output']).resolve()
    wav = Path(request['input']).resolve()
    assert output.is_relative_to(Path(request['output_root']).resolve())
    assert not output.is_relative_to(source) and sys.dont_write_bytecode
    sys.path.insert(0, str(source))
    import numpy as np
    from scipy.io import wavfile
    from phonetic_toolbox.models.config import EGGConfig
    from phonetic_toolbox.services.egg_service import EGGAnalysisService
    from phonetic_toolbox.core.egg.analysis import calculate_cq_sq, find_gci_goi_peak_min_criterion
    from phonetic_toolbox.core.acoustic.f0_praat import compute_praat_f0_track

    arrays = {}

    def pack(value, key):
        if dataclasses.is_dataclass(value):
            return {f.name: pack(getattr(value, f.name), key + '.' + f.name)
                    for f in dataclasses.fields(value)}
        if isinstance(value, np.ndarray):
            arrays[key] = value.copy()
            return {'array': key}
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return {'nonfinite': str(value)}
        if isinstance(value, (tuple, list)):
            return [pack(v, key + '.' + str(i)) for i, v in enumerate(value)]
        if isinstance(value, dict):
            return {k: pack(v, key + '.' + k) for k, v in value.items()}
        return value

    def call(fn, key):
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            try:
                value = fn()
                record = {'status': 'returned', 'value': pack(value, key)}
            except Exception as exc:
                value = None
                message = str(exc).replace(str(wav), '<input>').replace(str(source), '<v2>')
                record = {'status': 'error', 'type': type(exc).__name__, 'message': message}
        record['warnings'] = sorted(set(str(w.message) for w in observed))
        return value, record

    service = EGGAnalysisService()
    config = EGGConfig()
    report = {'schema_version': 1, 'case_id': request['case_id'], 'input_sha256': sha(wav),
              'service_defaults': dataclasses.asdict(config)}
    result, report['load'] = call(lambda: service.load_file(str(wav), config,
                                                          flip_channels=request.get('flip_channels', False)), 'load')
    stopped = threading.Event()
    stopped.set()
    report['cancellation'] = {
        'load': pack(service.load_file(str(wav), config, cancel_event=stopped), 'cancel.load'),
        'events': pack(find_gci_goi_peak_min_criterion(np.zeros(100), 16000, cancel_event=stopped), 'cancel.events'),
    }
    report['analytic'] = pack({'cq': calculate_cq_sq([0, .01, .02], [.006, .016], [.002, .014])}, 'analytic')
    report['analytic']['boundaries'] = pack(calculate_cq_sq([0, 1, 2, 3], [.05, 1.95], [.02, 1.3]), 'analytic.boundaries')
    report['analytic']['multiple_peaks'] = pack(calculate_cq_sq([0, 1], [.6], [.2, .3]), 'analytic.multiple_peaks')
    if result is not None:
        report['input_geometry'] = {'sample_rate_hz': result.fs, 'sample_count': len(result.time_vector),
                                    'sample_duration_s': len(result.time_vector)/result.fs,
                                    'last_sample_time_s': result.file_duration,
                                    'flip_channels': request.get('flip_channels', False)}
        variants = []
        methods = [('slope', 'scale', True)]
        if request.get('matrix'):
            methods = [(gci, goi, auto) for gci in ('slope', 'scale')
                       for goi in ('slope', 'scale') for auto in (False, True)]
        for index, (gci, goi, auto) in enumerate(methods):
            cfg = dataclasses.replace(config, gci_method=gci, goi_method=goi, auto_prominence=auto)
            current = service.analyze_events(copy.deepcopy(result), cfg)
            prefix = f'variant.{index}'
            variant = {'config': dataclasses.asdict(cfg),
                       'events': pack((current.gci_times, current.goi_times, current.peak_times), prefix + '.events'),
                       'gci_f0': pack((current.gci_f0_times, current.gci_f0_values), prefix + '.f0'),
                       'global_cq': pack(calculate_cq_sq(current.gci_times, current.goi_times, current.peak_times), prefix + '.global_cq'),
                       'roi': {}}
            duration = result.file_duration
            bounds = {'middle': (min(.2, duration/4), min(.4, duration*.75)),
                      'start': (0, min(.1, duration)),
                      'end': (max(0, duration-.05), duration+.05),
                      'outside': (duration+1, duration+2)}
            for label, (start, end) in bounds.items():
                modes = {}
                for raw in (False, True):
                    mode = 'raw' if raw else 'filtered'
                    _, cq = call(lambda: service.calculate_cq_sq_segment(current, start, end, cfg, raw),
                                 prefix + '.' + label + '.' + mode + '.cq')
                    _, events = call(lambda: service.get_events_segment(current, start, end, cfg, raw),
                                     prefix + '.' + label + '.' + mode + '.events')
                    modes[mode] = {'bounds_s': [start, end], 'cq': cq, 'events': events}
                variant['roi'][label] = modes
            variants.append(variant)
        report['variants'] = variants
        if request.get('matrix'):
            from phonetic_toolbox.core.signals.filters import apply_highpass_filter, apply_lowpass_filter
            report['filter_probes'] = {}
            for cutoff in (1, 25, 50, -1, result.fs):
                for label, fn in [('high', apply_highpass_filter), ('low', apply_lowpass_filter)]:
                    key = f'{label}_{cutoff}'
                    _, report['filter_probes'][key] = call(lambda: fn(result.egg_signal_raw[:4000], cutoff, result.fs), 'filter.' + key)
            report['ignored_parameters'] = {}
            for criterion, minimum in ((.25, 50), (.75, 50), (.25, 200)):
                key = f'{criterion}_{minimum}'
                report['ignored_parameters'][key] = pack(find_gci_goi_peak_min_criterion(
                    result.egg_signal_processed, result.fs, min_f0=minimum, criterion_level=criterion,
                    gci_method='scale', goi_method='scale'), 'ignored.' + key)
            movement = copy.deepcopy(result)
            movement.audio_f0_times = np.array([0, .01, .02, .03, .14, .15, .16])
            movement.audio_f0_values = np.array([100, 130, 100, 140, 400, 100, np.nan])
            service.detect_glottal_movement(movement)
            report['analytic']['movement'] = pack(movement.glottal_movement_events, 'analytic.movement')
            report['analytic']['gci_f0'] = {}
            for label, times in [('low', [0, .02, .04, .06]), ('outlier', [0, .005, .010, .03, .035, .04]),
                                 ('duplicate', [0, .01, .01, .02])]:
                movement.gci_times = times
                service._calculate_gci_f0(movement)
                report['analytic']['gci_f0'][label] = pack((movement.gci_f0_times, movement.gci_f0_values), 'analytic.f0.' + label)
        cfg = dataclasses.replace(config, goi_method='scale')
        current = service.analyze_events(copy.deepcopy(result), cfg)
        service.calculate_praat_f0(current)
        report['praat'] = {'legacy': pack({'times': current.audio_f0_times, 'values': current.audio_f0_values}, 'praat.legacy')}
        audio_path = output / 'normalized-audio.wav'
        wavfile.write(audio_path, result.fs, result.audio_signal)
        _, report['praat']['actual'] = call(lambda: compute_praat_f0_track(audio_path, 10, 75, 600, 'ac'), 'praat.actual')
        service.detect_glottal_movement(current)
        report['glottal'] = pack(current.glottal_movement_events, 'glottal')
        length = min(len(result.audio_signal), int(.12 * result.fs))
        inverse_audio = result.audio_signal[:length]
        gci = np.asarray([t for t in current.gci_times if t < length / result.fs])
        report['inverse'] = {}
        for label, order, events in [('auto', None, gci), ('explicit', 12, gci), ('no_gci', None, np.array([])),
                                     ('too_large', length+1, gci)]:
            _, report['inverse'][label] = call(lambda: service.apply_simplified_cp_inverse_filtering(
                inverse_audio, result.fs, events, lp_order=order), 'inverse.' + label)
        if request.get('gui'):
            capture_gui(request, output, wav, service, current, report, pack, arrays)

    report['arrays'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                          'sha256': hashlib.sha256(v.tobytes()).hexdigest(),
                          'nan_count': int(np.isnan(v).sum()), 'inf_count': int(np.isinf(v).sum())}
                        for k, v in sorted(arrays.items())}
    report['modules_sha256'] = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('phonetic_toolbox') and path:
            path = Path(path).resolve()
            assert path.is_relative_to(source), 'Reference module escaped original v2'
            report['modules_sha256'][name] = sha(path)
    write_json(output / 'result.json', report)
    np.savez_compressed(output / 'arrays.npz', **arrays)
    write_json(output / 'environment.json', {'python': sys.version,
        'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}})
    print(json.dumps({'id': request['case_id'], 'status': report['load']['status'], 'arrays': len(arrays)}), flush=True)


def capture_gui(request, output, wav, service, result, report, pack, arrays):
    import numpy as np
    import pandas as pd
    from PIL import Image
    from PyQt6.QtCore import QPoint
    from PyQt6.QtGui import QFont, QFontDatabase
    from PyQt6.QtWidgets import QApplication, QAbstractButton, QLineEdit
    from phonetic_toolbox.gui.widgets.egg_widget import EGGWidget
    from phonetic_toolbox.gui.dialogs.egg_batch_dialog import BatchWorker
    from phonetic_toolbox.models.config import EGGConfig

    app = QApplication.instance() or QApplication([])
    # The offscreen plugin has no system font fallback. Read the existing Windows font;
    # this is a test presentation override, not an edit to v2 or an installed dependency.
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 9))
    widget = EGGWidget()
    widget.resize(1600, 1000)
    widget.show()
    app.processEvents()
    report['ui'] = {'config': dataclasses.asdict(widget.config),
                    'timeline_window_s': widget.timeline_window_s, 'zoom_window_ms': widget.zoom_window_ms,
                    'geometry': {}, 'controls': []}
    for name in ('cq_canvas', 'spec_canvas', 'audio_zoom_canvas', 'egg_zoom_canvas', 'timeline_canvas',
                 'start_time_input', 'auto_prom_checkbox', 'highpass_slider'):
        child = getattr(widget, name)
        pos = child.mapTo(widget, QPoint(0, 0))
        report['ui']['geometry'][name] = [pos.x(), pos.y(), child.width(), child.height()]
    for child in widget.findChildren(QAbstractButton):
        report['ui']['controls'].append({'text': child.text(), 'checked': child.isChecked(), 'enabled': child.isEnabled()})
    report['ui']['inputs'] = [c.text() for c in widget.findChildren(QLineEdit)]
    widget.result = result
    widget.current_filepath = str(wav)
    widget.current_roi_start, widget.current_roi_duration = .1, .4
    widget.start_time_input.setText('0.1')
    widget.duration_input.setText('0.4')
    widget.show_f0 = widget.f0_corrected = True
    widget.show_f0_checkbox.blockSignals(True)
    widget.show_f0_checkbox.setChecked(True)
    widget.correct_f0_checkbox.blockSignals(True)
    widget.correct_f0_checkbox.setChecked(True)
    widget.set_theme(False)
    widget.update_roi_plots()
    widget.plot_timeline()
    widget.update_zoom_plots(.25)
    app.processEvents()
    widget.grab().save(str(output / 'v2-layout.png'))
    report['ui']['capture_scope'] = 'original QWidget offscreen, explicit existing Windows UI font, light plot theme; no full-app/device acceptance'
    report['spectrogram'] = {}
    for window in (5, 20, 50):
        widget.config.spec_window_ms = window
        widget.update_roi_plots()
        report['spectrogram'][str(window)] = [{'extent': list(im.get_extent()),
                                               'clim': list(im.get_clim()),
                                               'data': pack(np.asarray(im.get_array()), f'spectrum.{window}.{i}')}
                                              for i, im in enumerate(widget.spec_ax.images)]
    widget.config.spec_window_ms = 20

    def artifacts(folder, key):
        png = []
        for path in sorted(folder.glob('*.png')):
            with Image.open(path) as im:
                im.load()
                png.append({'name': path.name, 'size': list(im.size), 'dpi': list(im.info.get('dpi', ())),
                            'pixel_sha256': hashlib.sha256(im.tobytes()).hexdigest()})
        csv = sorted(folder.glob('*.csv'))
        record = {'png': png, 'csv_count': len(csv), 'files': sorted(p.name for p in folder.iterdir())}
        if len(csv) == 1:
            frame = pd.read_csv(csv[0])
            record.update(columns=list(frame.columns), rows=len(frame), csv_sha256=sha(csv[0]))
            arrays['export.' + key + '.csv'] = frame.select_dtypes(include='number').to_numpy()
        return record

    single = output / 'single'
    single.mkdir()
    widget._save_csv_data(.1, .5, str(single / 'DATA.csv'))
    widget._save_plots(.1, .5, *(str(single / (name + '.png')) for name in ('SPEC_F0', 'CQ_SQ', 'WAVEFORMS')))
    report['exports'] = {'single': artifacts(single, 'single')}
    inputs = output / 'batch-input'
    inputs.mkdir()
    shutil.copyfile(wav, inputs / 'sample.wav')
    for label, params in [('batch', {'generate_images': True}),
                          ('masked', {'silence_threshold': 1.0}), ('cancelled', {}),
                          ('no_praat', {'keep_f0': False}), ('no_gci_f0', {'keep_corr_f0': False}),
                          ('no_f0', {'keep_f0': False, 'keep_corr_f0': False})]:
        dest = output / label
        dest.mkdir()
        worker = BatchWorker(service, str(inputs), str(dest), EGGConfig(), params)
        if label == 'cancelled':
            worker.cancel()
        worker.run()
        report['exports'][label] = artifacts(dest, label)
    # Actual old per-file failure/continue behavior, using two owned synthetic inputs.
    (inputs / 'broken.wav').write_bytes(b'not a WAV')
    dest = output / 'errors'
    dest.mkdir()
    logs = []
    worker = BatchWorker(service, str(inputs), str(dest), EGGConfig(), {})
    worker.log.connect(logs.append)
    worker.run()
    report['exports']['errors'] = artifacts(dest, 'errors')
    report['exports']['errors']['failure_count'] = sum('失败:' in line for line in logs)
    widget.close()
    app.processEvents()


if __name__ == '__main__':
    main()

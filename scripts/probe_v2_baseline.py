"""Reference-environment behavioral probes; writes new synthetic exports only."""
import dataclasses
import json
import marshal
import sqlite3
import sys
import types
from pathlib import Path

from baseline_support import RECIPES, create_fixture, load_json, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert sys.dont_write_bytecode
    source = Path(load_json(ROOT / 'docs/baseline/local-evidence.json')['baseline']['root'])
    output = Path(sys.argv[1]).resolve()
    assert output.is_relative_to(ROOT / 'output/validation/p03')
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source))
    import numpy as np
    import pandas as pd
    from scipy.io.wavfile import read
    from phonetic_toolbox.models.config import AcousticConfig, EGGConfig
    from phonetic_toolbox.services.acoustic_service import AcousticAnalysisService, PARAMETER_MAPPING
    from phonetic_toolbox.services.egg_service import EGGAnalysisService
    from phonetic_toolbox.services.io.excel import get_fast_param_path

    def embedded(name):
        module = types.ModuleType('p03_embedded_' + name.rsplit('.', 1)[-1])
        sys.modules[module.__name__] = module
        code = marshal.loads((ROOT / 'output/validation/p03/executable' / (name + '.marshal')).read_bytes())
        exec(code, module.__dict__)
        return module

    config_module = embedded('phonetic_toolbox.models.config')
    assert dataclasses.asdict(config_module.AcousticConfig()) == dataclasses.asdict(AcousticConfig())
    assert dataclasses.asdict(config_module.EGGConfig()) == dataclasses.asdict(EGGConfig())
    service_module = embedded('phonetic_toolbox.services.acoustic_service')
    fixture = create_fixture(output, RECIPES[0])
    config = AcousticConfig(reaper_bin_path=str(source / 'phonetic_toolbox/core/acoustic/reaper.exe'))
    original = AcousticAnalysisService().analyze_file(str(fixture), config)
    extracted = service_module.AcousticAnalysisService().analyze_file(str(fixture), config)
    expected = original.to_dataframe()
    pd.testing.assert_frame_equal(expected, extracted.to_dataframe(), check_exact=True)

    exports = output / 'exports'
    exports.mkdir()
    xlsx = exports / 'synthetic.xlsx'
    AcousticAnalysisService().save_results(original, str(xlsx))
    database = get_fast_param_path(xlsx)
    assert sorted(p.name for p in exports.iterdir()) == sorted([xlsx.name, database.name])
    renamed = expected.rename(columns=PARAMETER_MAPPING)
    excel_frame = pd.read_excel(xlsx)
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        sqlite_frame = pd.read_sql_query('SELECT * FROM params ORDER BY Time_s', connection)
    for frame in [excel_frame, sqlite_frame]:
        assert list(frame.columns) == list(renamed.columns)
        assert np.array_equal(frame.isna().values, renamed.isna().values)
        np.testing.assert_array_equal(frame['Time_s'].values, renamed['Time_s'].values)
        np.testing.assert_allclose(frame.values, renamed.values, rtol=1e-12, atol=1e-12, equal_nan=True)

    egg_checks = []
    for case in load_json(ROOT / 'output/validation/p03/confirmed-egg.json')['cases']:
        path = Path(case['input'])
        before = sha(path)
        fs, data = read(path)
        normalized = data.astype(np.float32)
        if np.issubdtype(data.dtype, np.integer):
            normalized /= np.iinfo(data.dtype).max
        for channel in [0, 1]:
            peak = float(np.max(np.abs(normalized[:, channel])))
            if peak > 0:
                normalized[:, channel] = normalized[:, channel] / peak * 0.7
        result = EGGAnalysisService().load_file(str(path), EGGConfig(), flip_channels=case['flip_channels'])
        index = int(case['flip_channels'])
        np.testing.assert_array_equal(result.egg_signal_raw, normalized[:, index])
        np.testing.assert_array_equal(result.audio_signal, normalized[:, 1 - index])
        assert result.fs == fs and sha(path) == before
        egg_checks.append({'id': case['case_id'], 'input_sha256': before, 'channels': 2,
                           'egg_channel_zero_based': index, 'audio_channel_zero_based': 1 - index,
                           'exact_selected_channel_check': True})
    report = {'status': 'verified', 'embedded_config_defaults_exact': True,
        'embedded_acoustic_service_dataframe_exact': True,
        'scope': 'Extracted EXE config/service bytecode with original conda dependencies; not full frozen EXE runtime or GUI',
        'export': {'file_count': 2, 'rows': len(expected), 'columns': len(expected.columns),
                   'time_and_missing_masks_exact': True, 'numeric_rtol': 1e-12, 'numeric_atol': 1e-12},
        'egg_channel_checks': egg_checks}
    write_json(output / 'probe-summary.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

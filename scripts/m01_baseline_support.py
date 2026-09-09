"""M01-A capture recipes only; no production scientific implementation."""
SETTINGS = {
    'silence_threshold': 0.2, 'energy_window_ms': 30.0, 'frameshift_ms': 10.0,
    'windowsize_ms': 200.0, 'smooth_win_size': 3, 'lip_smooth_win_size': 3,
    'only_voiced': False, 'n_periods': 4, 'num_formants': 6, 'max_formant': 5000.0,
    'min_f0': 75.0, 'max_f0': 500.0, 'reaper_hilbert': False, 'reaper_no_highpass': True,
}


def cases():
    result = [{'id': name, 'selection': selection} for name, selection in [
        ('SERVICE-NONE', None), ('GUI-ALL', 'all'), ('SERVICE-EMPTY', []),
        ('SELECT-PRAAT', ['pF0']), ('SELECT-REAPER', ['rF0']),
        ('SELECT-INTENSITY', ['Intensity']), ('SELECT-ENERGY', ['Energy'])]]
    result += [{'id':'ASSOCIATED', 'selection':'all', 'associations':True, 'export':True},
               {'id':'FORMULA-EXPORT', 'selection':'all', 'associations':True, 'export':True, 'formula':True},
               {'id':'EMPTY-EXPORT', 'empty':True, 'export':True},
               {'id':'REAPER-FAILURE', 'fault':'reaper_invalid_argument'},
               {'id':'IRAPT-FAILURE', 'fault':'irapt_raises'},
               {'id':'LIP-TIME', 'mode':'lip'}, {'id':'GUI-CONTROLS', 'mode':'controls'}]
    result += [{'id':'SETTING-'+key, 'setting':key, 'config':{key:value}, 'associations':True}
               for key,value in SETTINGS.items()]
    return result

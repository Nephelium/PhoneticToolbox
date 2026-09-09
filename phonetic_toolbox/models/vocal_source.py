"""User-facing source controls, in SI-derived units (Pa, mm, mm²)."""
SOURCE_PRESETS = {
    'voiced': {'mode':'voiced','pressure_pa':800.,'opening_mm':.2,'posterior_gap_mm2':3.,'vibration':1.},
    'voiceless': {'mode':'voiceless','pressure_pa':800.,'opening_mm':1.2,'posterior_gap_mm2':6.,'vibration':0.},
    # A non-vibrating membranous glottis with a posterior gap. Approximation,
    # not a calibrated reconstruction of an individual whispering larynx.
    'whisper': {'mode':'whisper','pressure_pa':800.,'opening_mm':.05,'posterior_gap_mm2':4.,'vibration':0.},
}
SOURCE_LIMITS = {'pressure_pa':(0.,1600.),'opening_mm':(0.,3.),'posterior_gap_mm2':(0.,25.),'vibration':(0.,1.)}

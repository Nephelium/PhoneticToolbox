from PhoneticToolbox.models.state import AppState


def test_defaults_match_matlab_init():
    s = AppState()
    assert s.windowsize == 25
    assert s.frameshift == 5
    assert s.preemphasis == 0.96
    assert s.lpcOrder == 12
    assert s.F0Praatmin == 40
    assert s.F0Praatmax == 500

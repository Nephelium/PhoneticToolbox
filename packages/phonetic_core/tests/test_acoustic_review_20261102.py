import numpy as np
import pytest
from phonetic_core.acoustic.energy import compute_energy
from phonetic_core.acoustic.corrections import iseli_correction, compute_H1H2_H2H4_corrected


def test_unit_pressure_and_unit_sine_reference():
    fs=16000;track=np.ones(100)
    constant=compute_energy(np.ones(fs),fs,10,track)
    np.testing.assert_allclose(constant,20*np.log10(1/20e-6),rtol=0,atol=1e-12)
    y=np.sin(2*np.pi*200*np.arange(fs)/fs)
    sine=compute_energy(y,fs,10,track)
    np.testing.assert_allclose(sine,10*np.log10(.5/(20e-6)**2),rtol=0,atol=1e-10)


def test_zero_power_is_missing_and_negative_db_is_valid():
    assert np.isnan(compute_energy(np.zeros(1600),16000,10,np.ones(10))).all()
    np.testing.assert_allclose(compute_energy(np.full(1600,2e-6),16000,10,np.ones(10)),-20,rtol=0,atol=1e-12)


@pytest.mark.parametrize('bandwidth',[np.nan,0,-1,np.inf])
def test_invalid_bandwidth_is_not_zero_correction(bandwidth):
    assert np.isnan(iseli_correction(200,500,bandwidth,16000))
    args=[np.array([v]) for v in (20,15,10,200,500,1500)]
    result=compute_H1H2_H2H4_corrected(*args[:3],16000,*args[3:],B1=np.array([bandwidth]),B2=np.array([100.]))
    assert all(np.isnan(value).all() for value in result.values())


def test_valid_correction_matches_pole_transfer_ratio():
    f,fx,bw,fs=200.,500.,80.,16000
    r=np.exp(-np.pi*bw/fs)
    coefficients=[1,-2*r*np.cos(2*np.pi*fx/fs),r*r]
    response=np.polyval(coefficients,np.exp(2j*np.pi*f/fs))
    dc=np.polyval(coefficients,1)
    expected=20*np.log10(abs(dc/response))
    assert iseli_correction(f,fx,bw,fs)==pytest.approx(expected,abs=1e-10)

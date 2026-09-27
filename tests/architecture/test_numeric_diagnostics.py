"""P11 diagnostic error summaries must not hide sign/mask/shape differences."""
import importlib.util
from pathlib import Path
import pytest

np=pytest.importorskip('numpy')
spec=importlib.util.spec_from_file_location('numeric_diagnostics',Path(__file__).resolve().parents[2]/'scripts/diagnose_numeric_compatibility.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_signed_zero_and_infinity_are_not_hidden_by_zero_finite_error():
    result=module.compare_arrays(np.array([-0.,np.inf,np.nan]),np.array([0.,-np.inf,np.nan]))
    assert not result['exact'] and result['max_abs']==0
    assert result['nan_mask_equal'] and not result['positive_inf_mask_equal'] and not result['negative_inf_mask_equal']


def test_known_difference_and_shape_mismatch():
    result=module.compare_arrays(np.array([0.,3.,np.nan]),np.array([0.,2.,np.nan]))
    assert result['changed_finite']==1 and result['max_abs']==1 and result['max_rel_nonzero']==.5
    result=module.compare_arrays(np.array([1.,2.]),np.array([1.]))
    assert not result['exact'] and 'max_abs' not in result


def test_object_addresses_are_rejected_and_dtype_matters():
    with pytest.raises(ValueError,match='Object arrays'):module.compare_arrays(np.array(None),np.array(None))
    assert not module.compare_arrays(np.array([1],dtype='int32'),np.array([1],dtype='float64'))['exact']


def test_cross_platform_input_identifiers_preserve_hashes():
    assert module.normalized_hashes({'tests\\fixture.npz':'abc'}) == {'tests/fixture.npz':'abc'}
    with pytest.raises(ValueError,match='Ambiguous'):
        module.normalized_hashes({'tests\\fixture.npz':'abc','tests/fixture.npz':'def'})

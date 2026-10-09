"""A18/A19: long-source contract and strict bounded-input decisions."""
import pytest
from ptb_api.egg_models import EggTaskConfig
from ptb_worker.egg_child import MAX_SECONDS,MAX_SAMPLES

def test_measured_long_input_limits():
 assert MAX_SECONDS==120
 assert MAX_SAMPLES==5_760_000

def test_micro_center_can_reach_long_recording_tail():
 assert EggTaskConfig(mode='preview',roi_start=119.5,roi_end=120,micro_center=119.75).micro_center==119.75
 assert EggTaskConfig(mode='preview',micro_center=1799.75).micro_center==1799.75
 with pytest.raises(ValueError):EggTaskConfig(mode='preview',micro_center=1800.001)

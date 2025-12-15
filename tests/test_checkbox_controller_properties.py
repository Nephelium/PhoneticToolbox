"""
Property-based tests for CheckboxController in Whisper Transcription Tool.

Tests checkbox mutual exclusion and auto-selection properties.
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_transcription.checkbox_controller import CheckboxController


# ============================================================================
# Mock Checkbox for Testing
# ============================================================================

class MockCheckbox:
    """Mock checkbox class for testing without PyQt6 dependency."""
    
    def __init__(self, checked: bool = False):
        self._checked = checked
    
    def isChecked(self) -> bool:
        return self._checked
    
    def setChecked(self, checked: bool) -> None:
        self._checked = checked


# ============================================================================
# Strategies for generating test data
# ============================================================================

# Strategy for checkbox states
checkbox_state = st.booleans()

# Strategy for initial checkbox configurations
checkbox_config = st.tuples(checkbox_state, checkbox_state, checkbox_state)


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(
    initial_save_text=checkbox_state,
    initial_convert_pinyin=checkbox_state,
    initial_save_pinyin=checkbox_state
)
@settings(max_examples=100)
def test_checkbox_mutual_exclusion_after_check_save_pinyin(
    initial_save_text: bool,
    initial_convert_pinyin: bool,
    initial_save_pinyin: bool
):
    """
    **Feature: whisper-transcription, Property 8: Checkbox Mutual Exclusion**
    **Validates: Requirements 6.2, 6.3**
    
    For any state of the checkbox controller, after checking checkbox 3 (Save Pinyin as Lab),
    checkbox 1 (Save Text as Lab) should be unchecked.
    """
    # Create mock checkboxes with initial states
    cb_save_text = MockCheckbox(initial_save_text)
    cb_convert_pinyin = MockCheckbox(initial_convert_pinyin)
    cb_save_pinyin = MockCheckbox(initial_save_pinyin)
    
    # Create controller
    controller = CheckboxController(cb_save_text, cb_convert_pinyin, cb_save_pinyin)
    
    # Simulate checking checkbox 3
    cb_save_pinyin.setChecked(True)
    controller.on_save_pinyin_changed(True)
    
    # Verify mutual exclusion: checkbox 1 should be unchecked
    assert not cb_save_text.isChecked(), \
        "Checkbox 1 (Save Text) should be unchecked when checkbox 3 (Save Pinyin) is checked"
    
    # Verify checkbox 3 is checked
    assert cb_save_pinyin.isChecked(), \
        "Checkbox 3 (Save Pinyin) should remain checked"


@given(
    initial_save_text=checkbox_state,
    initial_convert_pinyin=checkbox_state,
    initial_save_pinyin=checkbox_state
)
@settings(max_examples=100)
def test_checkbox_mutual_exclusion_after_check_save_text(
    initial_save_text: bool,
    initial_convert_pinyin: bool,
    initial_save_pinyin: bool
):
    """
    **Feature: whisper-transcription, Property 8: Checkbox Mutual Exclusion**
    **Validates: Requirements 6.2, 6.3**
    
    For any state of the checkbox controller, after checking checkbox 1 (Save Text as Lab),
    checkbox 3 (Save Pinyin as Lab) should be unchecked.
    """
    # Create mock checkboxes with initial states
    cb_save_text = MockCheckbox(initial_save_text)
    cb_convert_pinyin = MockCheckbox(initial_convert_pinyin)
    cb_save_pinyin = MockCheckbox(initial_save_pinyin)
    
    # Create controller
    controller = CheckboxController(cb_save_text, cb_convert_pinyin, cb_save_pinyin)
    
    # Simulate checking checkbox 1
    cb_save_text.setChecked(True)
    controller.on_save_text_changed(True)
    
    # Verify mutual exclusion: checkbox 3 should be unchecked
    assert not cb_save_pinyin.isChecked(), \
        "Checkbox 3 (Save Pinyin) should be unchecked when checkbox 1 (Save Text) is checked"
    
    # Verify checkbox 1 is checked
    assert cb_save_text.isChecked(), \
        "Checkbox 1 (Save Text) should remain checked"


@given(
    initial_save_text=checkbox_state,
    initial_save_pinyin=checkbox_state
)
@settings(max_examples=100)
def test_mutual_exclusion_pure_function_check_save_text(
    initial_save_text: bool,
    initial_save_pinyin: bool
):
    """
    **Feature: whisper-transcription, Property 8: Checkbox Mutual Exclusion**
    **Validates: Requirements 6.2, 6.3**
    
    For any initial state, applying 'check_save_text' action should result in
    checkbox 1 being checked and checkbox 3 being unchecked.
    """
    new_save_text, new_save_pinyin = CheckboxController.apply_mutual_exclusion(
        initial_save_text, initial_save_pinyin, 'check_save_text'
    )
    
    # Checkbox 1 should be checked
    assert new_save_text is True, "Checkbox 1 should be checked after check_save_text action"
    
    # Checkbox 3 should be unchecked (mutual exclusion)
    assert new_save_pinyin is False, "Checkbox 3 should be unchecked after check_save_text action"


@given(
    initial_save_text=checkbox_state,
    initial_save_pinyin=checkbox_state
)
@settings(max_examples=100)
def test_mutual_exclusion_pure_function_check_save_pinyin(
    initial_save_text: bool,
    initial_save_pinyin: bool
):
    """
    **Feature: whisper-transcription, Property 8: Checkbox Mutual Exclusion**
    **Validates: Requirements 6.2, 6.3**
    
    For any initial state, applying 'check_save_pinyin' action should result in
    checkbox 3 being checked and checkbox 1 being unchecked.
    """
    new_save_text, new_save_pinyin = CheckboxController.apply_mutual_exclusion(
        initial_save_text, initial_save_pinyin, 'check_save_pinyin'
    )
    
    # Checkbox 1 should be unchecked (mutual exclusion)
    assert new_save_text is False, "Checkbox 1 should be unchecked after check_save_pinyin action"
    
    # Checkbox 3 should be checked
    assert new_save_pinyin is True, "Checkbox 3 should be checked after check_save_pinyin action"


@given(config=checkbox_config)
@settings(max_examples=100)
def test_checkboxes_never_both_checked_after_any_action(config: tuple):
    """
    **Feature: whisper-transcription, Property 8: Checkbox Mutual Exclusion**
    **Validates: Requirements 6.2, 6.3**
    
    For any state of the checkbox controller, checkboxes 1 and 3 should never
    both be checked simultaneously after any controller action.
    """
    initial_save_text, initial_convert_pinyin, initial_save_pinyin = config
    
    # Create mock checkboxes
    cb_save_text = MockCheckbox(initial_save_text)
    cb_convert_pinyin = MockCheckbox(initial_convert_pinyin)
    cb_save_pinyin = MockCheckbox(initial_save_pinyin)
    
    controller = CheckboxController(cb_save_text, cb_convert_pinyin, cb_save_pinyin)
    
    # Test action: check save_pinyin
    cb_save_pinyin.setChecked(True)
    controller.on_save_pinyin_changed(True)
    
    # Verify mutual exclusion
    assert not (cb_save_text.isChecked() and cb_save_pinyin.isChecked()), \
        "Checkboxes 1 and 3 should never both be checked"
    
    # Reset and test the other action
    cb_save_text = MockCheckbox(initial_save_text)
    cb_convert_pinyin = MockCheckbox(initial_convert_pinyin)
    cb_save_pinyin = MockCheckbox(initial_save_pinyin)
    
    controller = CheckboxController(cb_save_text, cb_convert_pinyin, cb_save_pinyin)
    
    # Test action: check save_text
    cb_save_text.setChecked(True)
    controller.on_save_text_changed(True)
    
    # Verify mutual exclusion
    assert not (cb_save_text.isChecked() and cb_save_pinyin.isChecked()), \
        "Checkboxes 1 and 3 should never both be checked"


# ============================================================================
# Property 9: Checkbox Auto-Selection Tests
# ============================================================================

@given(
    initial_save_text=checkbox_state,
    initial_convert_pinyin=checkbox_state,
    initial_save_pinyin=checkbox_state
)
@settings(max_examples=100)
def test_checkbox_auto_selection_when_save_pinyin_checked(
    initial_save_text: bool,
    initial_convert_pinyin: bool,
    initial_save_pinyin: bool
):
    """
    **Feature: whisper-transcription, Property 9: Checkbox Auto-Selection**
    **Validates: Requirements 5.3, 6.1**
    
    For any state where checkbox 3 (Save Pinyin as Lab) is checked,
    checkbox 2 (Convert to Pinyin) should also be checked.
    """
    # Create mock checkboxes with initial states
    cb_save_text = MockCheckbox(initial_save_text)
    cb_convert_pinyin = MockCheckbox(initial_convert_pinyin)
    cb_save_pinyin = MockCheckbox(initial_save_pinyin)
    
    # Create controller
    controller = CheckboxController(cb_save_text, cb_convert_pinyin, cb_save_pinyin)
    
    # Simulate checking checkbox 3
    cb_save_pinyin.setChecked(True)
    controller.on_save_pinyin_changed(True)
    
    # Verify auto-selection: checkbox 2 should be checked
    assert cb_convert_pinyin.isChecked(), \
        "Checkbox 2 (Convert to Pinyin) should be auto-selected when checkbox 3 (Save Pinyin) is checked"


@given(initial_convert_pinyin=checkbox_state)
@settings(max_examples=100)
def test_auto_selection_pure_function_when_save_pinyin_true(initial_convert_pinyin: bool):
    """
    **Feature: whisper-transcription, Property 9: Checkbox Auto-Selection**
    **Validates: Requirements 5.3, 6.1**
    
    For any initial state of checkbox 2, when checkbox 3 is checked,
    checkbox 2 should become checked.
    """
    # When save_pinyin is True, convert_pinyin should always be True
    result = CheckboxController.apply_auto_selection(
        convert_pinyin_checked=initial_convert_pinyin,
        save_pinyin_checked=True
    )
    
    assert result is True, \
        "Checkbox 2 should be True when checkbox 3 is checked"


@given(initial_convert_pinyin=checkbox_state)
@settings(max_examples=100)
def test_auto_selection_pure_function_when_save_pinyin_false(initial_convert_pinyin: bool):
    """
    **Feature: whisper-transcription, Property 9: Checkbox Auto-Selection**
    **Validates: Requirements 5.3, 6.1**
    
    For any initial state of checkbox 2, when checkbox 3 is not checked,
    checkbox 2 should retain its original state.
    """
    # When save_pinyin is False, convert_pinyin should keep its original state
    result = CheckboxController.apply_auto_selection(
        convert_pinyin_checked=initial_convert_pinyin,
        save_pinyin_checked=False
    )
    
    assert result == initial_convert_pinyin, \
        "Checkbox 2 should retain its original state when checkbox 3 is not checked"


@given(config=checkbox_config)
@settings(max_examples=100)
def test_valid_state_after_save_pinyin_action(config: tuple):
    """
    **Feature: whisper-transcription, Property 9: Checkbox Auto-Selection**
    **Validates: Requirements 5.3, 6.1**
    
    For any initial state, after checking checkbox 3, the controller state
    should be valid (checkbox 2 must be checked if checkbox 3 is checked).
    """
    initial_save_text, initial_convert_pinyin, initial_save_pinyin = config
    
    # Create mock checkboxes
    cb_save_text = MockCheckbox(initial_save_text)
    cb_convert_pinyin = MockCheckbox(initial_convert_pinyin)
    cb_save_pinyin = MockCheckbox(initial_save_pinyin)
    
    controller = CheckboxController(cb_save_text, cb_convert_pinyin, cb_save_pinyin)
    
    # Simulate checking checkbox 3
    cb_save_pinyin.setChecked(True)
    controller.on_save_pinyin_changed(True)
    
    # Verify the state is valid
    assert controller.is_valid_state(), \
        "Controller state should be valid after checking checkbox 3"
    
    # Specifically verify the auto-selection property
    save_text, convert_pinyin, save_pinyin = controller.get_state()
    if save_pinyin:
        assert convert_pinyin, \
            "If checkbox 3 is checked, checkbox 2 must also be checked"


@given(config=checkbox_config)
@settings(max_examples=100)
def test_is_valid_state_property(config: tuple):
    """
    **Feature: whisper-transcription, Property 9: Checkbox Auto-Selection**
    **Validates: Requirements 5.3, 6.1**
    
    For any state where checkbox 3 is checked and checkbox 2 is not checked,
    is_valid_state should return False.
    """
    initial_save_text, initial_convert_pinyin, initial_save_pinyin = config
    
    # Create mock checkboxes
    cb_save_text = MockCheckbox(initial_save_text)
    cb_convert_pinyin = MockCheckbox(initial_convert_pinyin)
    cb_save_pinyin = MockCheckbox(initial_save_pinyin)
    
    controller = CheckboxController(cb_save_text, cb_convert_pinyin, cb_save_pinyin)
    
    # Get current state
    save_text, convert_pinyin, save_pinyin = controller.get_state()
    
    # Check validity
    is_valid = controller.is_valid_state()
    
    # If checkbox 3 is checked but checkbox 2 is not, state should be invalid
    if save_pinyin and not convert_pinyin:
        assert not is_valid, \
            "State should be invalid when checkbox 3 is checked but checkbox 2 is not"
    
    # If both checkboxes 1 and 3 are checked, state should be invalid
    if save_text and save_pinyin:
        assert not is_valid, \
            "State should be invalid when both checkboxes 1 and 3 are checked"

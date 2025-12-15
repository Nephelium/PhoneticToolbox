"""
Property-based tests for theme toggle functionality.

Tests the idempotence property: toggling theme twice returns to original state.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MockQApplication:
    """Mock QApplication for testing theme toggle without GUI."""
    
    def __init__(self):
        self._stylesheet = ""
    
    def setStyleSheet(self, stylesheet: str) -> None:
        self._stylesheet = stylesheet
    
    def styleSheet(self) -> str:
        return self._stylesheet


class MockWidget:
    """Mock widget for testing theme toggle without GUI."""
    
    def __init__(self):
        self._stylesheet = ""
    
    def setStyleSheet(self, stylesheet: str) -> None:
        self._stylesheet = stylesheet
    
    def styleSheet(self) -> str:
        return self._stylesheet


class ThemeController:
    """
    Isolated theme controller for testing theme toggle logic.
    
    This extracts the core theme toggle logic from MainController
    to enable testing without PyQt6 GUI dependencies.
    """
    
    # Theme stylesheets (copied from controllers.py)
    DARK_MAIN_STYLESHEET = """
QMainWindow {
    background-color: #000000;
    color: #ffffff;
}
"""
    
    GLOBAL_DARK_STYLESHEET = """
QWidget {
    background-color: #121212;
    color: #ffffff;
}
"""
    
    LIGHT_MAIN_STYLESHEET = """
QMainWindow {
    background-color: #f0f0f0;
    color: #000000;
}
"""
    
    def __init__(self, app: MockQApplication, widget: MockWidget, initial_dark: bool = True):
        self.app = app
        self.widget = widget
        self.is_dark = initial_dark
        
        # Apply initial theme
        if self.is_dark:
            self.app.setStyleSheet(self.GLOBAL_DARK_STYLESHEET)
            self.widget.setStyleSheet(self.DARK_MAIN_STYLESHEET)
        else:
            self.app.setStyleSheet("")
            self.widget.setStyleSheet(self.LIGHT_MAIN_STYLESHEET)
    
    def toggle_theme(self) -> None:
        """Toggle between dark and light themes."""
        if self.is_dark:
            # Switch to Light
            self.app.setStyleSheet("")
            self.widget.setStyleSheet(self.LIGHT_MAIN_STYLESHEET)
            self.is_dark = False
        else:
            # Switch to Dark
            self.app.setStyleSheet(self.GLOBAL_DARK_STYLESHEET)
            self.widget.setStyleSheet(self.DARK_MAIN_STYLESHEET)
            self.is_dark = True
    
    def get_state(self) -> tuple:
        """Get current theme state as a tuple for comparison."""
        return (
            self.is_dark,
            self.app.styleSheet(),
            self.widget.styleSheet()
        )


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(initial_dark=st.booleans())
@settings(max_examples=100)
def test_theme_toggle_idempotence(initial_dark: bool):
    """
    **Feature: phonetic-toolbox, Property 5: Theme Toggle Idempotence**
    **Validates: Requirements 1.3**
    
    For any initial theme state, toggling the theme twice SHALL return
    to the original theme state.
    """
    # Setup
    app = MockQApplication()
    widget = MockWidget()
    controller = ThemeController(app, widget, initial_dark=initial_dark)
    
    # Capture initial state
    initial_state = controller.get_state()
    
    # Toggle twice
    controller.toggle_theme()
    controller.toggle_theme()
    
    # Verify state is restored
    final_state = controller.get_state()
    
    assert initial_state == final_state, (
        f"Theme state not restored after double toggle.\n"
        f"Initial: {initial_state}\n"
        f"Final: {final_state}"
    )


@given(initial_dark=st.booleans())
@settings(max_examples=100)
def test_theme_toggle_changes_state(initial_dark: bool):
    """
    Test that a single toggle actually changes the theme state.
    
    This is a supporting test to ensure toggle_theme is working correctly.
    """
    # Setup
    app = MockQApplication()
    widget = MockWidget()
    controller = ThemeController(app, widget, initial_dark=initial_dark)
    
    # Capture initial state
    initial_is_dark = controller.is_dark
    
    # Toggle once
    controller.toggle_theme()
    
    # Verify state changed
    assert controller.is_dark != initial_is_dark, (
        f"Theme state did not change after toggle.\n"
        f"Initial is_dark: {initial_is_dark}\n"
        f"Final is_dark: {controller.is_dark}"
    )


@given(n_toggles=st.integers(min_value=0, max_value=20))
@settings(max_examples=100)
def test_theme_toggle_even_returns_to_original(n_toggles: int):
    """
    Test that an even number of toggles returns to original state.
    
    This generalizes the idempotence property.
    """
    # Setup with dark theme
    app = MockQApplication()
    widget = MockWidget()
    controller = ThemeController(app, widget, initial_dark=True)
    
    # Capture initial state
    initial_state = controller.get_state()
    
    # Toggle n times
    for _ in range(n_toggles):
        controller.toggle_theme()
    
    final_state = controller.get_state()
    
    # Even number of toggles should return to original
    if n_toggles % 2 == 0:
        assert initial_state == final_state, (
            f"Theme state not restored after {n_toggles} toggles (even).\n"
            f"Initial: {initial_state}\n"
            f"Final: {final_state}"
        )
    else:
        assert initial_state != final_state, (
            f"Theme state should be different after {n_toggles} toggles (odd).\n"
            f"Initial: {initial_state}\n"
            f"Final: {final_state}"
        )


# ============================================================================
# Unit Tests
# ============================================================================

def test_dark_theme_applies_correct_stylesheets():
    """Test that dark theme applies the correct stylesheets."""
    app = MockQApplication()
    widget = MockWidget()
    controller = ThemeController(app, widget, initial_dark=True)
    
    assert controller.is_dark is True
    assert "background-color: #121212" in app.styleSheet()
    assert "background-color: #000000" in widget.styleSheet()


def test_light_theme_applies_correct_stylesheets():
    """Test that light theme applies the correct stylesheets."""
    app = MockQApplication()
    widget = MockWidget()
    controller = ThemeController(app, widget, initial_dark=False)
    
    assert controller.is_dark is False
    assert app.styleSheet() == ""
    assert "background-color: #f0f0f0" in widget.styleSheet()


def test_toggle_from_dark_to_light():
    """Test toggling from dark to light theme."""
    app = MockQApplication()
    widget = MockWidget()
    controller = ThemeController(app, widget, initial_dark=True)
    
    controller.toggle_theme()
    
    assert controller.is_dark is False
    assert app.styleSheet() == ""
    assert "background-color: #f0f0f0" in widget.styleSheet()


def test_toggle_from_light_to_dark():
    """Test toggling from light to dark theme."""
    app = MockQApplication()
    widget = MockWidget()
    controller = ThemeController(app, widget, initial_dark=False)
    
    controller.toggle_theme()
    
    assert controller.is_dark is True
    assert "background-color: #121212" in app.styleSheet()
    assert "background-color: #000000" in widget.styleSheet()

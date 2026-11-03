"""The native reader's verified fullscreen transition is the only exception."""
import unittest

from .core import ManualError
from .register_localized_captures import validate_window


class CaptureWindowTests(unittest.TestCase):
    def test_ordinary_capture_still_requires_actual_maximization(self):
        self.assertFalse(validate_window({'maximized': True}, 'm18', 'other-image'))
        with self.assertRaises(ManualError):
            validate_window({'maximized': False}, 'm18', 'other-image')

    def test_native_fullscreen_requires_complete_enter_and_exit_evidence(self):
        evidence = dict(maximized=False, fullscreen=True, documentFullscreen=True,
                        wasMaximized=True, fullscreenExitRestoredMaximized=True)
        asset = 'm18-r3-reader-fullscreen-da8aa864'
        self.assertTrue(validate_window(evidence, 'm18', asset))
        for key in ('documentFullscreen', 'wasMaximized', 'fullscreenExitRestoredMaximized'):
            with self.subTest(missing=key), self.assertRaises(ManualError):
                validate_window({k: v for k, v in evidence.items() if k != key}, 'm18', asset)
        for chapter, identity in [('m17', asset), ('m18', 'other-image')]:
            with self.subTest(chapter=chapter, asset=identity), self.assertRaises(ManualError):
                validate_window(evidence, chapter, identity)
        with self.assertRaises(ManualError):
            validate_window({**evidence, 'maximized': True}, 'm18', asset)


if __name__ == '__main__':
    unittest.main()

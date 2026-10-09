"""A resized Qt surface must not uncover an old Chromium viewport."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QColor, QImage
from types import SimpleNamespace
from ptb_desktop.presentation import FirstMaximizePresentation


class DelayedPage:
    def __init__(self):
        self.callbacks = []
        self.loading = False

    def isLoading(self):
        return self.loading

    def zoomFactor(self):
        return 1

    def runJavaScript(self, code, *args):
        callback = args[-1]
        self.callbacks.append(callback)


@pytest.fixture
def surface():
    app = QApplication.instance() or QApplication([])
    view = QWidget()
    view.resize(400, 300)
    view.show()
    app.processEvents()
    page = DelayedPage()
    guard = FirstMaximizePresentation(view, page)
    guard.set_color(QColor('#f9f9f9'))
    yield view, page, guard
    guard.finish()
    view.close()
    view.deleteLater()
    QCoreApplication.sendPostedEvents()


def test_cover_is_shown_before_native_resize(surface):
    view, page, guard = surface
    guard.prepare()
    assert guard.cover.isVisible()
    original = guard.cover.pixmap().size()
    view.resize(800, 600)
    assert guard.cover.size() == view.size()
    assert guard.cover.pixmap().size() == original


def test_old_document_callback_cannot_release_new_viewport(surface):
    view, page, guard = surface
    guard.prepare()
    guard.begin()
    old = page.callbacks[-1]
    view.resize(800, 600)
    old(True)
    assert guard.active
    assert not guard.document_ready


def test_restore_cancels_late_callback(surface):
    view, page, guard = surface
    guard.prepare()
    guard.begin()
    old = page.callbacks[-1]
    guard.finish()
    old(True)
    assert not guard.active
    assert not guard.cover.isVisible()


def test_second_maximize_and_regular_resize_do_not_capture(surface):
    view, page, guard = surface
    guard.prepare()
    guard.begin()
    guard.finish()
    guard.prepare()
    view.resize(600, 400)
    assert not guard.active
    assert not guard.cover.isVisible()


def test_no_post_resize_readback_when_native_hook_missing(surface):
    view, page, guard = surface
    view.resize(800, 600)
    view.grab = lambda: pytest.fail('A post-resize GPU readback can capture black pixels')
    guard.begin()
    assert guard.cover.isVisible()
    assert guard.cover.text()


def test_black_or_wrong_size_frame_cannot_uncover_surface(surface):
    view, page, guard = surface
    guard.prepare()
    guard.begin()
    guard.document_ready = True
    picture = QImage(400, 300, QImage.Format.Format_RGB32)
    picture.fill(QColor('black'))
    guard.quick = SimpleNamespace(size=view.size, devicePixelRatioF=lambda:1, grabFramebuffer=lambda:picture, update=lambda:None)
    guard.check_picture(guard.generation)
    assert guard.active and guard.frames == 0
    picture.fill(QColor('#f9f9f9'))
    view.resize(800, 600)
    guard.document_ready = True
    guard.check_picture(guard.generation)
    assert guard.active and guard.frames == 0


def test_only_two_valid_current_frames_uncover_surface(surface):
    view, page, guard = surface
    guard.prepare()
    guard.begin()
    guard.document_ready = True
    picture = QImage(400, 300, QImage.Format.Format_RGB32)
    picture.fill(QColor('#1e1e1e'))
    guard.quick = SimpleNamespace(size=view.size, devicePixelRatioF=lambda:1, grabFramebuffer=lambda:picture, update=lambda:None)
    guard.check_picture(guard.generation)
    assert guard.active and guard.frames == 1
    guard.check_picture(guard.generation)
    assert not guard.active and guard.frames == 2


def test_timeout_does_not_expose_unrendered_page(surface):
    view, page, guard = surface
    guard.prepare()
    guard.begin()
    guard.timed_out()
    assert guard.active and guard.cover.isVisible()
    assert guard.cover.text()


def test_interrupted_first_attempt_is_rearmed(surface):
    view, page, guard = surface
    guard.prepare()
    guard.begin()
    old = page.callbacks[-1]
    guard.cancel()
    guard.prepare()
    guard.begin()
    old(True)
    assert guard.active and not guard.document_ready

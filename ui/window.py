"""Desktop overlay window hosting the Live2D avatar via transparent QWebEngineView.

Features
--------
- Frameless, always-on-top, transparent window.
- Geometry from config (WINDOW_W × WINDOW_H), capped at 90 % of available screen.
- Default position: centre of the screen under the mouse cursor.
- Position/size persisted with QSettings; restored only when fully on screen.
- Optional macOS NSWindow "all spaces" behaviour via pyobjc.
- Global hotkey (pynput, CENTER_HOTKEY) to re-centre (signal-safe, no direct Qt calls).
- System tray icon with Bring to centre / Reset / Show+Hide / Quit.
- Keyboard shortcuts: arrows nudge, Shift+arrows large nudge, +/- resize, 0 reset, C centre.
- Safety-net timer every 5 s that re-centres the window if it leaves all screens.
- Diagnostic logging when DEBUG=True: screen list, window geometry, JS canvas info.
- bridge._page is wired here so bridge.py can call runJavaScript().
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from config import get_settings
from core.events import events
from ui.bridge import WebBridge

logger = logging.getLogger("anya.ui.window")

# ---------------------------------------------------------------------------
# Optional PySide6 imports
# ---------------------------------------------------------------------------
try:
    from PySide6.QtCore import (
        QPoint, QRect, QSettings, QSize, Qt, QTimer, QUrl, Signal,
    )
    from PySide6.QtGui import (
        QAction, QColor, QContextMenuEvent, QIcon, QKeyEvent, QMouseEvent,
    )
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    QWebEngineView = object  # type: ignore[misc,assignment]
    Signal = object          # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Settings key constants
# ---------------------------------------------------------------------------
_SETTINGS_ORG  = "Anya"
_SETTINGS_APP  = "AnyaCompanion"
_SETTINGS_GEO  = "window/geometry"   # QByteArray from saveGeometry()
_SETTINGS_POS  = "window/pos"        # QPoint fallback
_SETTINGS_SIZE = "window/size"       # QSize fallback


# ---------------------------------------------------------------------------
# Helper: find the screen under the mouse cursor (or primary screen)
# ---------------------------------------------------------------------------
def _screen_under_cursor() -> "QScreen":  # type: ignore[name-defined]
    """Return the QScreen that currently contains the mouse cursor."""
    from PySide6.QtGui import QCursor
    cursor_pos = QCursor.pos()
    for screen in QApplication.screens():
        if screen.geometry().contains(cursor_pos):
            return screen
    return QApplication.primaryScreen()


def _is_rect_visible(rect: "QRect") -> bool:  # type: ignore[name-defined]
    """Return True if *rect* is fully contained within some screen's availableGeometry."""
    for screen in QApplication.screens():
        if screen.availableGeometry().contains(rect):
            return True
    return False


def _log_screens() -> None:
    """Log diagnostic info about all connected screens."""
    for screen in QApplication.screens():
        geo = screen.geometry()
        avail = screen.availableGeometry()
        dpr = screen.devicePixelRatio()
        logger.debug(
            "Screen '%s': geo=%s avail=%s dpr=%.2f",
            screen.name(), geo, avail, dpr,
        )


# ---------------------------------------------------------------------------
# Main window class
# ---------------------------------------------------------------------------
class AnyaAvatarWindow(QWebEngineView):  # type: ignore[misc]
    """Frameless, transparent, always-on-top desktop overlay window for Anya avatar."""

    # Signal emitted from the pynput hotkey thread → handled on main thread.
    _center_requested: "Signal" = Signal()  # type: ignore[assignment]

    def __init__(self, debug_view: bool = False, reset_window: bool = False,
                 parent: Optional[object] = None) -> None:
        if not PYSIDE_AVAILABLE:
            raise RuntimeError("PySide6 is required to run AnyaAvatarWindow.")

        super().__init__(parent)
        self.settings_cfg = get_settings()
        self.debug_view = debug_view or self.settings_cfg.DEBUG

        self._drag_pos: Optional[QPoint] = None
        self._click_through: bool = False
        self._reset_window = reset_window

        self.bridge = WebBridge(self)

        self._init_window()
        self._init_webchannel()
        self._init_tray()
        self._init_hotkey()
        self._init_safety_timer()
        self._load_avatar()

        # Wire bridge._page so bridge.py can call runJavaScript()
        self.bridge._page = self.page()

        if self.debug_view:
            _log_screens()

    # ── Window initialisation ────────────────────────────────────────────────

    def _init_window(self) -> None:
        """Configure window flags, transparency, size, and position."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            # No Qt.Tool – Tool windows are hidden when app loses focus on macOS
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.page().setBackgroundColor(QColor(0, 0, 0, 0))

        # ── Compute target size ───────────────────────────────────────────────
        cfg_w = self.settings_cfg.WINDOW_W
        cfg_h = self.settings_cfg.WINDOW_H

        screen = _screen_under_cursor()
        avail  = screen.availableGeometry()
        max_w  = int(avail.width()  * 0.90)
        max_h  = int(avail.height() * 0.90)
        win_w  = min(cfg_w, max_w)
        win_h  = min(cfg_h, max_h)

        # ── Try to restore saved position / size ──────────────────────────────
        qs = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        restored = False

        if not self._reset_window:
            saved_geo = qs.value(_SETTINGS_GEO)
            if saved_geo is not None:
                ok = self.restoreGeometry(saved_geo)
                geo = self.geometry()
                if ok and _is_rect_visible(geo):
                    logger.debug("Restored window geometry: %s", geo)
                    restored = True
                else:
                    logger.debug("Saved geometry %s is off-screen – using default.", geo)

        if not restored:
            # Centre on the screen under the cursor.
            cx = avail.left() + (avail.width()  - win_w) // 2
            cy = avail.top()  + (avail.height() - win_h) // 2
            self.setGeometry(QRect(cx, cy, win_w, win_h))
            logger.debug("Default window geometry: x=%d y=%d w=%d h=%d", cx, cy, win_w, win_h)

        if self._reset_window:
            self._clear_saved_geometry()

        # Apply macOS "all spaces" behaviour.
        self._apply_macos_all_spaces()

    def _clear_saved_geometry(self) -> None:
        """Remove persisted window position/size from QSettings."""
        qs = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        qs.remove(_SETTINGS_GEO)
        qs.remove(_SETTINGS_POS)
        qs.remove(_SETTINGS_SIZE)
        qs.sync()
        logger.info("Saved window geometry cleared.")

    def _save_geometry(self) -> None:
        """Persist current window geometry to QSettings."""
        qs = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        qs.setValue(_SETTINGS_GEO, self.saveGeometry())
        qs.sync()

    def _apply_macos_all_spaces(self) -> None:
        """On macOS, pin the NSWindow to all Spaces and above fullscreen apps."""
        if not self.settings_cfg.SHOW_ON_ALL_SPACES:
            return
        try:
            import AppKit  # type: ignore[import]
            # NSWindowCollectionBehaviorCanJoinAllSpaces = 1 << 0  = 1
            # NSWindowCollectionBehaviorFullScreenAuxiliary = 1 << 8 = 256
            NSWindowCollectionBehaviorCanJoinAllSpaces   = 1
            NSWindowCollectionBehaviorFullScreenAuxiliary = 1 << 8
            behavior = (
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )
            ns_app = AppKit.NSApplication.sharedApplication()
            for win in ns_app.windows():
                win.setCollectionBehavior_(behavior)
            logger.debug("macOS NSWindow collection behaviour set (all spaces + fullscreen aux).")
        except Exception as exc:
            logger.warning("Could not set NSWindow collection behaviour: %s", exc)

    # ── Web channel ───────────────────────────────────────────────────────────

    def _init_webchannel(self) -> None:
        """Register WebBridge on QWebChannel."""
        self.channel = QWebChannel(self.page())
        self.channel.registerObject("bridge", self.bridge)
        self.page().setWebChannel(self.channel)

    # ── Avatar page ───────────────────────────────────────────────────────────

    def _load_avatar(self) -> None:
        """Navigate to the avatar HTML page."""
        endpoint = "debug.html" if self.debug_view else "index.html"
        url = f"http://127.0.0.1:{self.settings_cfg.SERVER_PORT}/{endpoint}"
        logger.info("Loading Live2D avatar viewport: %s", url)
        self.setUrl(QUrl(url))

    # ── System tray ───────────────────────────────────────────────────────────

    def _init_tray(self) -> None:
        """Create the system tray icon with action menu."""
        icon_path = str(
            Path(self.settings_cfg.ASSETS_DIR) / "model" / "icon.png"
        )
        icon = QIcon(icon_path) if Path(icon_path).exists() else QIcon()

        self._tray = QSystemTrayIcon(icon, parent=self)
        self._tray.setToolTip("Anya – Voice Companion")

        menu = QMenu()

        act_centre = QAction("Bring Anya to Centre", menu)
        act_centre.triggered.connect(self.centre_on_cursor_screen)
        menu.addAction(act_centre)

        act_reset = QAction("Reset Size and Position", menu)
        act_reset.triggered.connect(self._reset_size_and_position)
        menu.addAction(act_reset)

        menu.addSeparator()

        act_toggle = QAction("Show / Hide", menu)
        act_toggle.triggered.connect(self._toggle_visibility)
        menu.addAction(act_toggle)

        menu.addSeparator()

        act_quit = QAction("Quit Anya", menu)
        act_quit.triggered.connect(self.quit_app)
        menu.addAction(act_quit)

        self._tray.setContextMenu(menu)
        self._tray.show()

        # Double-click on tray icon → bring to centre
        self._tray.activated.connect(
            lambda reason: self.centre_on_cursor_screen()
            if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None
        )

    def _toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def _reset_size_and_position(self) -> None:
        """Clear saved geometry and move to default centred position."""
        self._clear_saved_geometry()
        screen = _screen_under_cursor()
        avail  = screen.availableGeometry()
        win_w  = min(self.settings_cfg.WINDOW_W, int(avail.width()  * 0.90))
        win_h  = min(self.settings_cfg.WINDOW_H, int(avail.height() * 0.90))
        cx = avail.left() + (avail.width()  - win_w) // 2
        cy = avail.top()  + (avail.height() - win_h) // 2
        self.setGeometry(QRect(cx, cy, win_w, win_h))
        self.raise_()
        self.activateWindow()
        logger.info("Window reset to default size and position.")

    # ── Global hotkey (pynput) ────────────────────────────────────────────────

    def _init_hotkey(self) -> None:
        """Register a global hotkey that emits _center_requested on the main thread."""
        self._center_requested.connect(self.centre_on_cursor_screen)
        hotkey_str = self.settings_cfg.CENTER_HOTKEY
        if not hotkey_str:
            return

        # Pre-warm keyboard layout on main thread to prevent macOS HIToolbox _dispatch_assert_queue_fail SIGTRAP
        try:
            from pynput import keyboard
            dummy = keyboard.Listener(on_press=lambda _: None, on_release=lambda _: None)
            dummy.start()
            dummy.stop()
        except Exception:
            pass

        def _start_listener() -> None:
            try:
                from pynput import keyboard  # type: ignore[import]

                def _on_activate() -> None:
                    # pynput callback runs in its own thread → only emit the signal.
                    self._center_requested.emit()

                listener = keyboard.GlobalHotKeys({hotkey_str: _on_activate})
                listener.start()
                logger.info("Global hotkey registered: %s", hotkey_str)
            except Exception as exc:
                if "input monitoring" in str(exc).lower() or "accessibility" in str(exc).lower():
                    logger.warning(
                        "Global hotkey disabled – Input Monitoring permission is missing. "
                        "Grant access in System Settings → Privacy → Input Monitoring."
                    )
                else:
                    logger.warning("Could not register global hotkey '%s': %s", hotkey_str, exc)

        t = threading.Thread(target=_start_listener, daemon=True, name="AnyaHotkey")
        t.start()

    # ── Safety-net timer ──────────────────────────────────────────────────────

    def _init_safety_timer(self) -> None:
        """Every 5 s, re-centre the window if it has strayed off all screens."""
        self._safety_timer = QTimer(self)
        self._safety_timer.setInterval(5000)
        self._safety_timer.timeout.connect(self._safety_check)
        self._safety_timer.start()

    def _safety_check(self) -> None:
        """Re-centre if the window is no longer fully inside any screen."""
        if not self.isVisible():
            return
        if not _is_rect_visible(self.geometry()):
            logger.warning(
                "Window is off-screen (%s) – re-centring.", self.geometry()
            )
            self.centre_on_cursor_screen()
            if self.settings_cfg.DEBUG:
                _log_screens()

    # ── Centre helper ─────────────────────────────────────────────────────────

    def centre_on_cursor_screen(self) -> None:
        """Move the window to the centre of the screen under the cursor."""
        screen = _screen_under_cursor()
        avail  = screen.availableGeometry()
        w = self.width()
        h = self.height()
        cx = avail.left() + (avail.width()  - w) // 2
        cy = avail.top()  + (avail.height() - h) // 2
        self.move(cx, cy)
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()
        self._save_geometry()
        logger.info("Window centred at (%d, %d) on screen '%s'.", cx, cy, screen.name())

    # ── Qt event overrides ────────────────────────────────────────────────────

    def showEvent(self, event: object) -> None:  # type: ignore[override]
        super().showEvent(event)  # type: ignore[arg-type]
        self.raise_()
        self.activateWindow()
        self._apply_macos_all_spaces()  # re-apply after show on macOS
        if self.settings_cfg.DEBUG:
            logger.debug("Window shown: geometry=%s", self.geometry())

    def moveEvent(self, event: object) -> None:  # type: ignore[override]
        super().moveEvent(event)  # type: ignore[arg-type]
        self._save_geometry()
        if self.settings_cfg.DEBUG:
            logger.debug("Window moved: geometry=%s", self.geometry())

    def resizeEvent(self, event: object) -> None:  # type: ignore[override]
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._save_geometry()

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        """Handle keyboard shortcuts for nudge, resize, centre, reset."""
        key  = event.key()
        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        nudge = 100 if shift else 20

        pos = self.pos()

        if key == Qt.Key.Key_Left:
            self.move(pos.x() - nudge, pos.y())
        elif key == Qt.Key.Key_Right:
            self.move(pos.x() + nudge, pos.y())
        elif key == Qt.Key.Key_Up:
            self.move(pos.x(), pos.y() - nudge)
        elif key == Qt.Key.Key_Down:
            self.move(pos.x(), pos.y() + nudge)
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._resize_by(1.10)
        elif key == Qt.Key.Key_Minus:
            self._resize_by(0.90)
        elif key == Qt.Key.Key_0:
            self._reset_size_and_position()
        elif key == Qt.Key.Key_C:
            self.centre_on_cursor_screen()
        elif key == Qt.Key.Key_Escape:
            pass  # intentionally does nothing
        else:
            super().keyPressEvent(event)

    def _resize_by(self, factor: float) -> None:
        """Resize by *factor*, capped at 90 % of available screen."""
        screen = _screen_under_cursor()
        avail  = screen.availableGeometry()
        new_w  = int(self.width()  * factor)
        new_h  = int(self.height() * factor)
        new_w  = min(new_w, int(avail.width()  * 0.90))
        new_h  = min(new_h, int(avail.height() * 0.90))
        self.resize(new_w, new_h)

    # ── Mouse drag ────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    # ── Right-click context menu ──────────────────────────────────────────────

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        menu = QMenu(self)

        menu.addAction("Bring Anya to Centre").triggered.connect(self.centre_on_cursor_screen)
        menu.addAction("Reset Size and Position").triggered.connect(self._reset_size_and_position)
        menu.addSeparator()

        click_through_act = QAction("Click-Through Mode", menu)
        click_through_act.setCheckable(True)
        click_through_act.setChecked(self._click_through)
        click_through_act.triggered.connect(lambda checked: self.set_click_through(checked))
        menu.addAction(click_through_act)

        menu.addSeparator()
        menu.addAction("Quit Anya").triggered.connect(self.quit_app)

        menu.exec(event.globalPos())

    # ── Public helpers ────────────────────────────────────────────────────────

    def set_click_through(self, enable: bool) -> None:
        """Toggle mouse click transparency."""
        self._click_through = enable
        logger.info("Click-through toggled: %s", enable)
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, enable)
        self.show()

    def set_expression(self, expression_name: str) -> None:
        """Dispatch expression change to JavaScript."""
        self.bridge.set_emotion({expression_name.lower(): 1.0})

    def clear_expression(self) -> None:
        """Dispatch expression reset to JavaScript."""
        self.bridge.set_emotion({})

    def quit_app(self) -> None:
        """Trigger graceful application shutdown."""
        logger.info("User requested quit.")
        self._tray.hide()
        events.quit_requested.emit()
        app = QApplication.instance()
        if app is not None:
            app.quit()

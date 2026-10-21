"""Local HTTP server serving web UI and model assets with correct MIME types."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
from typing import Optional
from urllib.parse import unquote, urlparse

from config import get_settings

logger = logging.getLogger("anya.ui.server")

# Explicit MIME mappings for Live2D and web client assets
MIME_MAP = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".model3.json": "application/json; charset=utf-8",
    ".exp3.json": "application/json; charset=utf-8",
    ".cdi3.json": "application/json; charset=utf-8",
    ".physics3.json": "application/json; charset=utf-8",
    ".moc3": "application/octet-stream",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".wav": "audio/wav",
    ".ico": "image/x-icon",
}


class AnyaRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler serving ui/web at / and assets at /assets/."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.settings = get_settings()
        self.web_dir = (self.settings.BASE_DIR / "ui" / "web").resolve()
        self.assets_dir = self.settings.ASSETS_DIR.resolve()
        super().__init__(*args, **kwargs)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        """Route GET requests to /api/config, /assets/, or ui/web/."""
        parsed_url = urlparse(self.path)
        path = unquote(parsed_url.path)

        # Config endpoint for client-side FPS and debug settings
        if path == "/api/config":
            payload = json.dumps({
                "fps_active": self.settings.FPS_ACTIVE,
                "fps_idle": self.settings.FPS_IDLE,
                "debug": self.settings.DEBUG,
                "server_port": self.settings.SERVER_PORT,
                "model_scale": self.settings.MODEL_SCALE,
                "model_offset_y": self.settings.MODEL_OFFSET_Y,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(payload)
            return

        super().do_GET()

    def translate_path(self, path: str) -> str:
        """Map URL path to physical filesystem location."""
        parsed_url = urlparse(path)
        clean_path = unquote(parsed_url.path)

        if clean_path.startswith("/assets/"):
            relative = clean_path[len("/assets/") :].lstrip("/")
            target = (self.assets_dir / relative).resolve()
            return str(target)

        # Default: serve from ui/web/
        relative = clean_path.lstrip("/")
        if not relative or clean_path == "/":
            relative = "index.html"
        target = (self.web_dir / relative).resolve()
        return str(target)

    def guess_type(self, path: str) -> str:
        """Determine accurate Content-Type header from file extension."""
        path_str = str(path).lower()
        for ext, mime in MIME_MAP.items():
            if path_str.endswith(ext):
                return mime
        return super().guess_type(path)

    def end_headers(self) -> None:
        """Append CORS headers to every response."""
        self.send_cors_headers()
        super().end_headers()

    def send_cors_headers(self) -> None:
        """Send permissive CORS headers for local client requests."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def log_message(self, format: str, *args: object) -> None:
        """Redirect request logging to Anya logger at DEBUG level."""
        logger.debug("%s - " + format, self.address_string(), *args)


class AnyaThreadingServer(ThreadingHTTPServer):
    """Threading HTTPServer with immediate socket address reuse enabled."""

    allow_reuse_address = True
    daemon_threads = True


_SERVER_INSTANCE: Optional[ThreadingHTTPServer] = None
_SERVER_THREAD: Optional[threading.Thread] = None


def start_server(host: str = "127.0.0.1", port: Optional[int] = None) -> ThreadingHTTPServer:
    """Start local ThreadingHTTPServer on host:port in a background daemon thread."""
    global _SERVER_INSTANCE, _SERVER_THREAD
    if _SERVER_INSTANCE is not None:
        return _SERVER_INSTANCE

    settings = get_settings()
    server_port = port if port is not None else settings.SERVER_PORT

    httpd = AnyaThreadingServer((host, server_port), AnyaRequestHandler)
    _SERVER_INSTANCE = httpd

    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="AnyaHTTPServer")
    _SERVER_THREAD = thread
    thread.start()

    logger.info("Serving Anya UI & Assets on http://%s:%d/ (daemon thread)", host, server_port)
    return httpd


def stop_server(httpd: Optional[ThreadingHTTPServer] = None) -> None:
    """Shut down running HTTP server instance."""
    global _SERVER_INSTANCE, _SERVER_THREAD
    server = httpd or _SERVER_INSTANCE
    if server is not None:
        logger.info("Stopping HTTP server...")
        server.shutdown()
        server.server_close()
        _SERVER_INSTANCE = None
        _SERVER_THREAD = None

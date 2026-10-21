"""Lazy model loader and resource manager with idle timeout and RAM telemetry."""

from dataclasses import dataclass
import gc
import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("anya.core.resources")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False


def get_ram_usage() -> Dict[str, float]:
    """Return process RSS in MB and system RAM usage percentage."""
    if PSUTIL_AVAILABLE and psutil is not None:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / (1024 * 1024)
        sys_ram_percent = psutil.virtual_memory().percent
        return {"rss_mb": round(rss_mb, 2), "sys_percent": float(sys_ram_percent)}

    # Fallback to standard library resource module if psutil is not installed
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS, ru_maxrss is in bytes; on Linux, in kilobytes
        import sys
        scale = 1.0 / (1024 * 1024) if sys.platform == "darwin" else 1.0 / 1024
        rss_mb = rusage.ru_maxrss * scale
        return {"rss_mb": round(rss_mb, 2), "sys_percent": 0.0}
    except Exception:
        return {"rss_mb": 0.0, "sys_percent": 0.0}


def log_ram_usage(action: str, model_name: str) -> None:
    """Emit a formatted log line recording RAM metrics after a load or unload event."""
    ram = get_ram_usage()
    logger.info(
        "[RAM Telemetry] %s '%s' -> Process RSS: %.1f MB | System RAM: %.1f%%",
        action,
        model_name,
        ram["rss_mb"],
        ram["sys_percent"],
    )


@dataclass
class ManagedResource:
    """Metadata and runtime state for a lazily loaded model."""

    name: str
    loader: Callable[[], Any]
    unloader: Optional[Callable[[Any], None]] = None
    timeout_s: float = 300.0
    instance: Optional[Any] = None
    timer: Optional[threading.Timer] = None


class ResourceManager:
    """Thread-safe lazy model holder enforcing idle unloading and memory tracking."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._resources: Dict[str, ManagedResource] = {}

    def register(
        self,
        name: str,
        loader: Callable[[], Any],
        unloader: Optional[Callable[[Any], None]] = None,
        timeout_s: float = 300.0,
    ) -> None:
        """Register a lazy resource loader with optional unloader and idle timeout."""
        with self._lock:
            # If an existing resource is loaded, unload it before re-registering
            if name in self._resources:
                self.unload(name)
            self._resources[name] = ManagedResource(
                name=name,
                loader=loader,
                unloader=unloader,
                timeout_s=max(0.01, float(timeout_s)),
            )
            logger.debug("Registered managed resource '%s' (idle timeout: %.1fs)", name, timeout_s)

    def is_loaded(self, name: str) -> bool:
        """Check if the given resource is currently loaded in memory."""
        with self._lock:
            res = self._resources.get(name)
            return res is not None and res.instance is not None

    def get(self, name: str) -> Any:
        """Fetch resource, lazily loading on first use and resetting idle countdown."""
        with self._lock:
            if name not in self._resources:
                raise KeyError(f"Resource '{name}' is not registered.")

            res = self._resources[name]

            # Cancel existing idle timer while in use
            if res.timer is not None:
                res.timer.cancel()
                res.timer = None

            if res.instance is None:
                logger.info("Lazily loading resource '%s'...", name)
                res.instance = res.loader()
                log_ram_usage("Loaded", name)

            # Schedule idle unload timer
            self._arm_idle_timer(res)
            return res.instance

    def touch(self, name: str) -> None:
        """Reset the idle timer for a loaded resource without reloading."""
        with self._lock:
            res = self._resources.get(name)
            if res and res.instance is not None:
                if res.timer is not None:
                    res.timer.cancel()
                    res.timer = None
                self._arm_idle_timer(res)

    def _arm_idle_timer(self, res: ManagedResource) -> None:
        """Arm a daemon timer to automatically unload after idle timeout."""
        def _on_timeout() -> None:
            logger.info("Idle timeout (%.1fs) expired for resource '%s'", res.timeout_s, res.name)
            self.unload(res.name)

        timer = threading.Timer(res.timeout_s, _on_timeout)
        timer.daemon = True
        res.timer = timer
        timer.start()

    def unload(self, name: str) -> bool:
        """Unload a specific resource, execute unloader hook, and record RAM."""
        with self._lock:
            res = self._resources.get(name)
            if res is None or res.instance is None:
                return False

            if res.timer is not None:
                res.timer.cancel()
                res.timer = None

            instance = res.instance
            res.instance = None

            try:
                if res.unloader is not None:
                    res.unloader(instance)
            except Exception as exc:
                logger.exception("Error executing unloader hook for '%s': %s", name, exc)
            finally:
                del instance
                gc.collect()

            log_ram_usage("Unloaded", name)
            return True

    def unload_all(self) -> None:
        """Unload all currently active resources."""
        with self._lock:
            for name in list(self._resources.keys()):
                self.unload(name)


resources = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """Return the global ResourceManager singleton."""
    return resources

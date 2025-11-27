"""
Performance profiler for face swap operations.
Tracks timing across all major operations to identify bottlenecks.
"""
import time
import threading
from typing import Dict, List, Optional
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

# Global profiler state
_timings: Dict[str, List[float]] = defaultdict(list)
_call_counts: Dict[str, int] = defaultdict(int)
_lock = threading.Lock()
_enabled = True
_start_time: Optional[float] = None


def enable_profiler():
    """Enable profiling."""
    global _enabled, _start_time
    _enabled = True
    _start_time = time.perf_counter()
    clear_stats()


def disable_profiler():
    """Disable profiling."""
    global _enabled
    _enabled = False


def clear_stats():
    """Clear all profiling statistics."""
    global _timings, _call_counts
    with _lock:
        _timings.clear()
        _call_counts.clear()


@contextmanager
def profile_section(name: str):
    """Context manager to profile a code section."""
    if not _enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        with _lock:
            _timings[name].append(elapsed)
            _call_counts[name] += 1


def profile_function(name: Optional[str] = None):
    """Decorator to profile a function."""
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return func(*args, **kwargs)

            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                with _lock:
                    _timings[func_name].append(elapsed)
                    _call_counts[func_name] += 1

        return wrapper
    return decorator


def record_time(name: str, elapsed_ms: float):
    """Manually record a timing."""
    if not _enabled:
        return
    with _lock:
        _timings[name].append(elapsed_ms)
        _call_counts[name] += 1


def get_stats() -> Dict[str, Dict]:
    """Get profiling statistics."""
    stats = {}
    with _lock:
        for name, times in _timings.items():
            if times:
                stats[name] = {
                    'calls': _call_counts[name],
                    'total_ms': sum(times),
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'last_ms': times[-1] if times else 0
                }
    return stats


def print_stats(sort_by: str = 'total_ms', top_n: int = 20):
    """Print profiling statistics to console."""
    stats = get_stats()
    if not stats:
        print("\n[PROFILER] No timing data collected.\n")
        return

    # Calculate total time
    total_time = sum(s['total_ms'] for s in stats.values())
    elapsed = (time.perf_counter() - _start_time) * 1000 if _start_time else total_time

    # Sort by specified field
    sorted_stats = sorted(stats.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)

    print("\n" + "=" * 100)
    print(f"{'PERFORMANCE PROFILER RESULTS':^100}")
    print(f"{'Total elapsed: ' + f'{elapsed:.1f}ms':^100}")
    print("=" * 100)
    print(f"{'Operation':<45} {'Calls':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'%':>6}")
    print("-" * 100)

    for name, data in sorted_stats[:top_n]:
        pct = (data['total_ms'] / elapsed * 100) if elapsed > 0 else 0
        print(f"{name:<45} {data['calls']:>8} {data['total_ms']:>12.2f} {data['avg_ms']:>10.2f} {data['min_ms']:>10.2f} {data['max_ms']:>10.2f} {pct:>5.1f}%")

    print("-" * 100)
    print(f"{'TOTAL':<45} {sum(s['calls'] for s in stats.values()):>8} {total_time:>12.2f}")
    print("=" * 100 + "\n")


def get_summary_line() -> str:
    """Get a single-line summary for live display."""
    stats = get_stats()
    if not stats:
        return ""

    key_ops = ['face_detection', 'face_swap', 'face_enhance', 'frame_read', 'frame_write']
    parts = []

    for op in key_ops:
        if op in stats:
            parts.append(f"{op.split('_')[-1][:4]}:{stats[op]['avg_ms']:.0f}ms")

    return " | ".join(parts)


# Convenience class for tracking FPS
class FPSTracker:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: List[float] = []
        self.last_time: Optional[float] = None

    def tick(self):
        """Call once per frame to track FPS."""
        now = time.perf_counter()
        if self.last_time is not None:
            self.frame_times.append(now - self.last_time)
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        self.last_time = now

    def get_fps(self) -> float:
        """Get current FPS."""
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def reset(self):
        """Reset FPS tracking."""
        self.frame_times.clear()
        self.last_time = None


# Global FPS tracker instance
fps_tracker = FPSTracker()

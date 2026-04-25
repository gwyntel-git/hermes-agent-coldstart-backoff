"""Retry utilities — jittered backoff for decorrelated retries.

Replaces fixed exponential backoff with jittered delays to prevent
thundering-herd retry spikes when multiple sessions hit the same
rate-limited provider concurrently.
"""

import random
import threading
import time

# Monotonic counter for jitter seed uniqueness within the same process.
# Protected by a lock to avoid race conditions in concurrent retry paths
# (e.g. multiple gateway sessions retrying simultaneously).
_jitter_counter = 0
_jitter_lock = threading.Lock()


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Compute a jittered exponential backoff delay.

    Args:
        attempt: 1-based retry attempt number.
        base_delay: Base delay in seconds for attempt 1.
        max_delay: Maximum delay cap in seconds.
        jitter_ratio: Fraction of computed delay to use as random jitter
            range.  0.5 means jitter is uniform in [0, 0.5 * delay].

    Returns:
        Delay in seconds: min(base * 2^(attempt-1), max_delay) + jitter.

    The jitter decorrelates concurrent retries so multiple sessions
    hitting the same provider don't all retry at the same instant.
    """
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2 ** exponent), max_delay)

    # Seed from time + counter for decorrelation even with coarse clocks.
    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)

    return delay + jitter


# ── Cold-start-aware extended backoff for overloaded providers ──────────
#
# Serverless GPU providers (Modal, RunPod, etc.) return 503 "Service
# Unavailable" while a GPU container cold-starts — typically 30–120
# seconds.  The default jittered_backoff (5s base, 120s cap) burns
# through retries too quickly: 5s → 10s → 20s → 40s → 80s → 120s
# only sums to ~275s of total wait, and the first 3 attempts all fire
# within 35s when the container needs 60–90s.
#
# overloaded_backoff uses a longer base (30s) and higher cap (300s)
# to survive cold-start windows: 30s → 60s → 120s → 240s → 300s,
# giving the provider ~5 minutes to come online with only 5 attempts.

_OVERLOADED_BASE_DELAY = 30.0   # seconds — long enough to miss a cold start
_OVERLOADED_MAX_DELAY = 300.0   # seconds — cap at 5 minutes
_OVERLOADED_JITTER_RATIO = 0.3  # lower jitter — more predictable waits


def overloaded_backoff(attempt: int) -> float:
    """Compute an extended backoff delay for overloaded/cold-start providers.

    Uses longer timing than :func:`jittered_backoff` because 503/529
    from serverless providers typically means a GPU container is spinning
    up (30–120s), not a transient blip that resolves in seconds.

    Args:
        attempt: 1-based retry attempt number.

    Returns:
        Delay in seconds with cold-start-appropriate timing.
    """
    return jittered_backoff(
        attempt,
        base_delay=_OVERLOADED_BASE_DELAY,
        max_delay=_OVERLOADED_MAX_DELAY,
        jitter_ratio=_OVERLOADED_JITTER_RATIO,
    )

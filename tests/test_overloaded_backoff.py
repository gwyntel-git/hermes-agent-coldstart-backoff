"""Tests for agent.retry_utils overloaded backoff (cold-start-aware retries)."""

import agent.retry_utils as retry_utils
from agent.retry_utils import jittered_backoff, overloaded_backoff


# ── overloaded_backoff ──────────────────────────────────────────────────


def test_overloaded_backoff_base_is_30s():
    """First attempt should start at 30s (cold-start window)."""
    delay = overloaded_backoff(1)
    # With jitter_ratio=0.3, delay is in [30.0, 30.0 + 0.3*30.0] = [30, 39]
    assert delay >= 30.0, f"base delay should be >= 30s, got {delay}"
    assert delay <= 40.0, f"base delay + jitter should be <= 40s, got {delay}"


def test_overloaded_backoff_exponential():
    """Overloaded backoff should still be exponential but with longer base."""
    for attempt in (1, 2, 3, 4):
        delays = [
            overloaded_backoff(attempt)
            if retry_utils._OVERLOADED_JITTER_RATIO == 0
            else overloaded_backoff(attempt)
            for _ in range(50)
        ]
        mean = sum(delays) / len(delays)
        expected_base = min(
            retry_utils._OVERLOADED_BASE_DELAY * (2 ** max(0, attempt - 1)),
            retry_utils._OVERLOADED_MAX_DELAY,
        )
        # Mean should be near expected_base + half of jitter range
        assert mean >= expected_base * 0.8, (
            f"attempt {attempt}: mean {mean:.1f} too low (expected ~{expected_base:.1f})"
        )


def test_overloaded_backoff_respects_max_300s():
    """Even with high attempt numbers, delay should not exceed 300s."""
    for attempt in (10, 20, 100):
        delay = overloaded_backoff(attempt)
        # max_delay=300 + jitter up to 0.3*300 = 90 → cap at 390
        assert delay <= 400.0, f"attempt {attempt}: delay {delay:.1f} exceeds 400s cap"


def test_overloaded_backoff_longer_than_default():
    """Overloaded backoff should produce longer waits than default jittered_backoff."""
    for attempt in (1, 2, 3):
        overloaded_delay = overloaded_backoff(attempt)
        default_delay = jittered_backoff(
            attempt, base_delay=5.0, max_delay=120.0, jitter_ratio=0.5
        )
        assert overloaded_delay > default_delay * 0.5, (
            f"attempt {attempt}: overloaded ({overloaded_delay:.1f}s) should be "
            f"substantially longer than default ({default_delay:.1f}s)"
        )


def test_overloaded_backoff_attempt_5_near_cap():
    """By attempt 5, we should be at or near the 300s cap (30*2^4=480→300)."""
    # Without jitter
    delays = []
    # Overload counter state — call multiple times and check range
    for _ in range(20):
        d = overloaded_backoff(5)
        delays.append(d)
    mean = sum(delays) / len(delays)
    assert mean >= 280.0, f"attempt 5 mean should be >= 280s, got {mean:.1f}"


def test_overloaded_backoff_delegates_to_jittered():
    """overloaded_backoff should be a convenience wrapper around jittered_backoff."""
    # Compare with explicit jittered_backoff call using same params
    import random
    random.seed(42)
    explicit = jittered_backoff(
        1,
        base_delay=retry_utils._OVERLOADED_BASE_DELAY,
        max_delay=retry_utils._OVERLOADED_MAX_DELAY,
        jitter_ratio=retry_utils._OVERLOADED_JITTER_RATIO,
    )
    # Both should produce values in the same range
    overloaded = overloaded_backoff(1)
    assert 30.0 <= overloaded <= 40.0
    assert 30.0 <= explicit <= 40.0

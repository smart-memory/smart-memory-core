"""Tests verifying call-time env reads in events.py and instrumentation.py."""
import os


def test_is_observability_enabled_reads_env_at_call_time():
    """events._is_observability_enabled() reflects env var changes made after import."""
    from smartmemory.observability.events import _is_observability_enabled

    old = os.environ.get("SMARTMEMORY_OBSERVABILITY")
    try:
        os.environ["SMARTMEMORY_OBSERVABILITY"] = "false"
        assert _is_observability_enabled() is False, (
            "events._is_observability_enabled() must return False when env var is false"
        )

        os.environ["SMARTMEMORY_OBSERVABILITY"] = "true"
        assert _is_observability_enabled() is True, (
            "events._is_observability_enabled() must return True when env var is true"
        )
    finally:
        if old is None:
            os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        else:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = old


def test_is_observability_enabled_default_true():
    """events._is_observability_enabled() returns True when env var is absent."""
    from smartmemory.observability.events import _is_observability_enabled

    old = os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
    try:
        assert _is_observability_enabled() is True, (
            "events._is_observability_enabled() must default to True"
        )
    finally:
        if old is not None:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = old


def test_instrumentation_observability_reads_env_at_call_time():
    """instrumentation._is_observability_enabled() reflects env var changes made after import."""
    from smartmemory.observability.instrumentation import _is_observability_enabled

    old = os.environ.get("SMARTMEMORY_OBSERVABILITY")
    try:
        os.environ["SMARTMEMORY_OBSERVABILITY"] = "false"
        assert _is_observability_enabled() is False, (
            "instrumentation._is_observability_enabled() must return False when env var is false"
        )

        os.environ["SMARTMEMORY_OBSERVABILITY"] = "true"
        assert _is_observability_enabled() is True
    finally:
        if old is None:
            os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        else:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = old

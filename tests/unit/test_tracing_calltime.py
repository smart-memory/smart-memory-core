"""Tests verifying that tracing._is_enabled() reads the env var at call time."""
import os


def test_is_enabled_reads_env_at_call_time():
    """_is_enabled() reflects env var changes made after import."""
    from smartmemory.observability.tracing import _is_enabled

    old = os.environ.get("SMARTMEMORY_OBSERVABILITY")
    try:
        os.environ["SMARTMEMORY_OBSERVABILITY"] = "false"
        assert _is_enabled() is False, (
            "_is_enabled() must return False when SMARTMEMORY_OBSERVABILITY=false"
        )

        os.environ["SMARTMEMORY_OBSERVABILITY"] = "true"
        assert _is_enabled() is True, (
            "_is_enabled() must return True when SMARTMEMORY_OBSERVABILITY=true"
        )
    finally:
        if old is None:
            os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        else:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = old


def test_is_enabled_default_true():
    """_is_enabled() returns True when SMARTMEMORY_OBSERVABILITY is not set."""
    from smartmemory.observability.tracing import _is_enabled

    old = os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
    try:
        assert _is_enabled() is True, (
            "_is_enabled() must default to True when env var is absent"
        )
    finally:
        if old is not None:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = old

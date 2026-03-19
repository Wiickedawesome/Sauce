"""
tests/test_utils.py — Tests for adapters/utils.py retry wrapper.
"""

from unittest.mock import MagicMock, patch

import pytest

from sauce.adapters.utils import _is_transient, call_with_retry

# ── _is_transient ─────────────────────────────────────────────────────────────


class TestIsTransient:
    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("connection refused"),
            TimeoutError("read timed out"),
            OSError("network unreachable"),
        ],
    )
    def test_network_exceptions_are_transient(self, exc: Exception) -> None:
        assert _is_transient(exc) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "HTTP 429 Too Many Requests",
            "502 Bad Gateway",
            "503 Service Unavailable",
            "504 Gateway Timeout",
            "connection reset by peer",
            "request timed out",
        ],
    )
    def test_message_markers_are_transient(self, msg: str) -> None:
        assert _is_transient(Exception(msg)) is True

    def test_auth_error_is_not_transient(self) -> None:
        assert _is_transient(Exception("403 Forbidden")) is False

    def test_validation_error_is_not_transient(self) -> None:
        assert _is_transient(ValueError("invalid symbol")) is False


# ── call_with_retry ───────────────────────────────────────────────────────────


class TestCallWithRetry:
    def test_succeeds_first_try(self) -> None:
        fn = MagicMock(return_value=42)
        assert call_with_retry(fn) == 42
        fn.assert_called_once()

    def test_passes_args_kwargs(self) -> None:
        fn = MagicMock(return_value="ok")
        call_with_retry(fn, "a", "b", key="val")
        fn.assert_called_once_with("a", "b", key="val")

    @patch("sauce.adapters.utils.time.sleep")
    def test_retries_on_transient_then_succeeds(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[ConnectionError("fail"), ConnectionError("fail"), "ok"])
        result = call_with_retry(fn)
        assert result == "ok"
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("sauce.adapters.utils.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[TimeoutError(), TimeoutError(), "ok"])
        call_with_retry(fn)
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0]  # base * 2^0, base * 2^1

    @patch("sauce.adapters.utils.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=ConnectionError("persistent"))
        with pytest.raises(ConnectionError, match="persistent"):
            call_with_retry(fn)
        assert fn.call_count == 3  # _MAX_RETRIES

    def test_permanent_error_raises_immediately(self) -> None:
        fn = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            call_with_retry(fn)
        fn.assert_called_once()  # No retry for non-transient

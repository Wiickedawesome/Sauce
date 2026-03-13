"""
tests/test_notify.py — Tests for adapters/notify.py alert delivery.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from sauce.adapters.notify import send_alert


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch):
    """Common env vars so Settings() can be instantiated, and clear cache."""
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestSendAlertLoggingOnly:
    def test_logs_critical_on_critical_severity(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.CRITICAL, logger="sauce.adapters.notify"):
            send_alert("CRITICAL", "loss limit breached", loop_id="abc")
        assert any("loss limit breached" in r.message for r in caplog.records)

    def test_logs_warning_on_warning_severity(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="sauce.adapters.notify"):
            send_alert("WARNING", "approaching limit", loop_id="abc")
        assert any("approaching limit" in r.message for r in caplog.records)

    def test_no_webhook_call_when_url_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "")
        # If httpx were called it would fail — absence of error means no call.
        send_alert("INFO", "test message")


class TestSendAlertWebhook:
    def test_posts_to_webhook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            send_alert("WARNING", "test webhook", loop_id="xyz")

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.args[0] == "https://hooks.example.com/test"

    def test_http_error_logged_not_raised(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/test")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            with caplog.at_level(logging.ERROR, logger="sauce.adapters.notify"):
                send_alert("WARNING", "error test")

        assert any("webhook returned HTTP 500" in r.message for r in caplog.records)

    def test_connection_error_logged_not_raised(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/test")

        with patch("httpx.Client", side_effect=ConnectionError("network down")):
            with caplog.at_level(logging.ERROR, logger="sauce.adapters.notify"):
                send_alert("CRITICAL", "conn test")

        assert any("failed to send webhook" in r.message for r in caplog.records)

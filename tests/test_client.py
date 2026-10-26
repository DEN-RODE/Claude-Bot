"""Tests for the exchange client, against a stubbed HTTP layer.

No network is touched. The point is to pin down the two behaviours that are
easy to get wrong and expensive to get wrong: that the signature covers exactly
the bytes that are sent, and that order placement is never retried.
"""

from __future__ import annotations

import hashlib
import hmac
from urllib.parse import parse_qsl, urlparse

import pytest

from claudebot.clients.asterdex import AsterdexClient, AsterdexError

API_KEY = "test-key"
API_SECRET = "test-secret"


class StubResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = str(payload)

    def json(self):
        return self._payload


class StubSession:
    """Records every request and returns queued responses."""

    def __init__(self):
        self.calls = []
        self.queue = []
        self.closed = False

    def mount(self, *_a, **_kw):
        pass

    def close(self):
        self.closed = True

    def request(self, method, url, headers=None, timeout=None, data=None):
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "timeout": timeout}
        )
        if self.queue:
            return self.queue.pop(0)
        return StubResponse({})


@pytest.fixture
def client(monkeypatch):
    read, write = StubSession(), StubSession()
    monkeypatch.setattr(
        "claudebot.clients.asterdex._read_only_session", lambda *a, **kw: read
    )
    monkeypatch.setattr("claudebot.clients.asterdex._write_session", lambda: write)
    c = AsterdexClient(API_KEY, API_SECRET)
    c._test_read, c._test_write = read, write
    return c


def _query_of(url: str) -> str:
    return urlparse(url).query


def test_credentials_are_required():
    with pytest.raises(ValueError):
        AsterdexClient("", API_SECRET)
    with pytest.raises(ValueError):
        AsterdexClient(API_KEY, "")


def test_signature_covers_exactly_the_query_that_is_sent(client):
    """Recompute the HMAC over the sent query and require a match.

    Signing a re-ordered copy of the parameters passes with two keys and fails
    with six, which is why this test uses a call with several.
    """
    client._test_write.queue.append(StubResponse({"orderId": 1}))
    client.place_market_order("BTCUSDT", "BUY", 0.01, reduce_only=True)

    query = _query_of(client._test_write.calls[0]["url"])
    signed_part, _, signature = query.rpartition("&signature=")

    expected = hmac.new(
        API_SECRET.encode(), signed_part.encode(), hashlib.sha256
    ).hexdigest()
    assert signature == expected


def test_signed_requests_carry_a_timestamp_and_recv_window(client):
    client._test_read.queue.append(StubResponse({"assets": []}))
    client.account()

    params = dict(parse_qsl(_query_of(client._test_read.calls[0]["url"])))
    assert "timestamp" in params
    assert params["recvWindow"] == "5000"
    assert "signature" in params


def test_unsigned_market_data_carries_no_signature(client):
    client._test_read.queue.append(StubResponse([]))
    client.klines("BTCUSDT", "1h", 10)

    params = dict(parse_qsl(_query_of(client._test_read.calls[0]["url"])))
    assert "signature" not in params
    assert params["symbol"] == "BTCUSDT"


def test_orders_go_through_the_non_retrying_session(client):
    """Writes must use the session with no Retry adapter mounted."""
    client._test_write.queue.append(StubResponse({"orderId": 7}))
    client.place_market_order("BTCUSDT", "BUY", 0.01)

    assert len(client._test_write.calls) == 1
    assert client._test_read.calls == []


def test_reads_go_through_the_retrying_session(client):
    client._test_read.queue.append(StubResponse([]))
    client.klines("BTCUSDT")

    assert len(client._test_read.calls) == 1
    assert client._test_write.calls == []


def test_a_failed_order_is_not_retried(client):
    """A gateway error on an order must surface, not trigger a second order."""
    client._test_write.queue.append(StubResponse({"msg": "gateway timeout"}, 504))
    with pytest.raises(AsterdexError):
        client.place_market_order("BTCUSDT", "BUY", 0.01)

    assert len(client._test_write.calls) == 1


def test_every_request_has_a_timeout(client):
    client._test_read.queue.append(StubResponse([]))
    client._test_write.queue.append(StubResponse({}))
    client.klines("BTCUSDT")
    client.cancel_all_orders("BTCUSDT")

    for session in (client._test_read, client._test_write):
        for call in session.calls:
            assert call["timeout"] is not None and call["timeout"] > 0


def test_api_key_header_is_sent(client):
    client._test_read.queue.append(StubResponse([]))
    client.klines("BTCUSDT")
    assert client._test_read.calls[0]["headers"]["X-MBX-APIKEY"] == API_KEY


def test_set_leverage_rejects_a_silent_mismatch(client):
    """Some venues return 200 without applying the change."""
    client._test_write.queue.append(StubResponse({"leverage": 20}))
    with pytest.raises(AsterdexError, match="exchange applied 20"):
        client.set_leverage("BTCUSDT", 3)


def test_set_leverage_accepts_a_matching_response(client):
    client._test_write.queue.append(StubResponse({"leverage": 3}))
    assert client.set_leverage("BTCUSDT", 3)["leverage"] == 3


def test_invalid_side_is_rejected_before_any_request(client):
    with pytest.raises(ValueError):
        client.place_market_order("BTCUSDT", "LONG", 0.01)
    assert client._test_write.calls == []


def test_positions_filters_out_flat_entries(client):
    client._test_read.queue.append(
        StubResponse(
            [
                {"symbol": "BTCUSDT", "positionAmt": "0.0"},
                {"symbol": "ETHUSDT", "positionAmt": "1.5"},
            ]
        )
    )
    positions = client.positions()
    assert [p["symbol"] for p in positions] == ["ETHUSDT"]


def test_ohlcv_splits_klines_into_float_series(client):
    client._test_read.queue.append(
        StubResponse(
            [
                [1, "10", "12", "9", "11", "100"],
                [2, "11", "13", "10", "12", "150"],
            ]
        )
    )
    series = client.ohlcv("BTCUSDT")
    assert series["closes"] == [11.0, 12.0]
    assert series["highs"] == [12.0, 13.0]
    assert series["volumes"] == [100.0, 150.0]


def test_http_errors_raise_asterdex_error(client):
    client._test_read.queue.append(StubResponse({"code": -1121}, 400))
    with pytest.raises(AsterdexError, match="400"):
        client.account()


def test_context_manager_closes_sessions(monkeypatch):
    read, write = StubSession(), StubSession()
    monkeypatch.setattr(
        "claudebot.clients.asterdex._read_only_session", lambda *a, **kw: read
    )
    monkeypatch.setattr("claudebot.clients.asterdex._write_session", lambda: write)
    with AsterdexClient(API_KEY, API_SECRET):
        pass
    assert read.closed and write.closed

"""HTTP client for a Binance-Futures-compatible perpetuals API.

Three things in here are not decoration.

**The signature covers exactly the bytes that are sent.** The query string is
built once, signed, and then appended to the URL verbatim. The obvious
alternative, signing ``urlencode(sorted(params))`` and handing the dict to the
HTTP library, is wrong whenever the library serialises in a different order, and
it fails silently: requests with two parameters work, requests with six are
rejected as badly signed, and the bug looks like an intermittent auth problem.

**Order placement is never retried automatically.** Transient-error retries are
fine for reads and catastrophic for writes: a gateway timeout returned *after*
the exchange has accepted an order turns one position into two. Reads get a
retrying session, writes get a single attempt and an explicit error.

**Every request has a timeout.** A socket read with no timeout is a bot that
stops trading at an unpredictable moment and never recovers.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://fapi.asterdex.com"
DEFAULT_TIMEOUT = 10.0

#: Milliseconds the exchange will accept between our timestamp and its clock.
#: Explicit because the server default is short and clock drift on a long-running
#: machine otherwise produces rejected orders with an unhelpful error.
DEFAULT_RECV_WINDOW = 5000


class AsterdexError(RuntimeError):
    """An API call failed."""


def _read_only_session(retries: int = 2) -> requests.Session:
    """A session that retries idempotent reads on transient gateway errors."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=(502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _write_session() -> requests.Session:
    """A session with no retries at all. See the module docstring."""
    return requests.Session()


class AsterdexClient:
    """Thin synchronous client. Wrap calls in ``asyncio.to_thread`` from async code."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        recv_window: int = DEFAULT_RECV_WINDOW,
    ) -> None:
        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret are required")
        self.api_key = api_key
        self._api_secret = api_secret.encode("utf-8")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.recv_window = recv_window
        self._read_session = _read_only_session()
        self._write_session = _write_session()

    # -- signing ---------------------------------------------------------

    def _signed_query(self, params: dict[str, Any]) -> str:
        """Return the exact query string to send, signature included."""
        payload = dict(params)
        payload["timestamp"] = int(time.time() * 1000)
        payload["recvWindow"] = self.recv_window
        query = urlencode(payload)
        signature = hmac.new(
            self._api_secret, query.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"{query}&signature={signature}"

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = False,
    ) -> Any:
        params = params or {}
        if signed:
            url = f"{self.base_url}{path}?{self._signed_query(params)}"
            body = None
        else:
            url = f"{self.base_url}{path}?{urlencode(params)}" if params else (
                f"{self.base_url}{path}"
            )
            body = None

        session = self._read_session if method == "GET" else self._write_session
        try:
            response = session.request(
                method,
                url,
                headers=self._headers,
                timeout=self.timeout,
                data=body,
            )
        except requests.RequestException as exc:
            raise AsterdexError(f"{method} {path} failed: {exc}") from exc

        if not response.ok:
            raise AsterdexError(
                f"{method} {path} returned {response.status_code}: {response.text[:200]}"
            )
        return response.json()

    # -- market data -----------------------------------------------------

    def klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> list[list]:
        """Return raw OHLCV candles, oldest first."""
        return self._request(
            "GET",
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )

    def ohlcv(
        self, symbol: str, interval: str = "1h", limit: int = 500
    ) -> dict[str, list[float]]:
        """Return klines split into named float series ready for the indicators."""
        raw = self.klines(symbol, interval, limit)
        return {
            "timestamps": [int(k[0]) for k in raw],
            "opens": [float(k[1]) for k in raw],
            "highs": [float(k[2]) for k in raw],
            "lows": [float(k[3]) for k in raw],
            "closes": [float(k[4]) for k in raw],
            "volumes": [float(k[5]) for k in raw],
        }

    def mark_price(self, symbol: str) -> float:
        data = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(data["markPrice"])

    # -- account ---------------------------------------------------------

    def account(self) -> dict:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def positions(self) -> list[dict]:
        """Open positions only, filtered from the full position list."""
        data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        return [p for p in data if float(p.get("positionAmt", 0)) != 0]

    def open_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/openOrders", params, signed=True)

    # -- trading ---------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage and verify the exchange actually applied it.

        Some venues return 200 while leaving leverage unchanged when a position
        is already open on the symbol. Trusting the status code there means
        trading at a leverage you did not choose, so the response is checked.
        """
        result = self._request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": leverage},
            signed=True,
        )
        applied = int(result.get("leverage", 0))
        if applied != leverage:
            raise AsterdexError(
                f"{symbol}: requested leverage {leverage}, exchange applied {applied}"
            )
        return result

    def place_market_order(
        self, symbol: str, side: str, quantity: float, reduce_only: bool = False
    ) -> dict:
        """Place a market order. Not retried; see the module docstring."""
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_stop_order(
        self, symbol: str, side: str, stop_price: float, quantity: float
    ) -> dict:
        """Place a reduce-only stop market order."""
        return self._request(
            "POST",
            "/fapi/v1/order",
            {
                "symbol": symbol,
                "side": side,
                "type": "STOP_MARKET",
                "stopPrice": stop_price,
                "quantity": quantity,
                "reduceOnly": "true",
            },
            signed=True,
        )

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._request(
            "DELETE",
            "/fapi/v1/order",
            {"symbol": symbol, "orderId": order_id},
            signed=True,
        )

    def cancel_all_orders(self, symbol: str) -> dict:
        return self._request(
            "DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True
        )

    def close(self) -> None:
        self._read_session.close()
        self._write_session.close()

    def __enter__(self) -> "AsterdexClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

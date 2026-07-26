"""Proxy configuration and connectivity checks.

These helpers configure GitHub-only subprocesses and the verified downloader's
explicit fallback route.  The downloader sends GitHub through the proxy,
attempts Hugging Face directly before retrying transport failures through the
proxy, and keeps unrelated hosts direct.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_PROXY_URL = "http://172.23.240.1:8080"
DEFAULT_NO_PROXY = "localhost,127.0.0.1,::1"


class ProxyConfigurationError(ValueError):
    """Raised when a configured proxy URL is malformed."""


class ProxyConnectivityError(ConnectionError):
    """Raised when the configured proxy cannot be reached."""


def _first_nonempty(environment: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = environment.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _validate_proxy_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ProxyConfigurationError(
            f"proxy URL must use http or https, got {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise ProxyConfigurationError("proxy URL has no hostname")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ProxyConfigurationError(f"invalid proxy port in {value!r}") from error
    return value


@dataclass(frozen=True, slots=True)
class ProxySettings:
    """Normalized HTTP proxy settings.

    ``TWEN_PROXY_URL`` or the explicit ``proxy_url`` argument is the convenient
    way to override both protocols.  Individual conventional environment
    variables remain supported when separate proxies are required.
    """

    http_proxy: str = DEFAULT_PROXY_URL
    https_proxy: str = DEFAULT_PROXY_URL
    no_proxy: str = DEFAULT_NO_PROXY

    def __post_init__(self) -> None:
        if self.http_proxy:
            _validate_proxy_url(self.http_proxy)
        if self.https_proxy:
            _validate_proxy_url(self.https_proxy)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        proxy_url: str | None = None,
    ) -> ProxySettings:
        env = os.environ if environment is None else environment
        common = proxy_url or _first_nonempty(env, "TWEN_PROXY_URL")
        if common:
            http_proxy = https_proxy = common
        else:
            http_proxy = _first_nonempty(env, "HTTP_PROXY", "http_proxy")
            https_proxy = _first_nonempty(env, "HTTPS_PROXY", "https_proxy")
            http_proxy = http_proxy or DEFAULT_PROXY_URL
            https_proxy = https_proxy or http_proxy
        no_proxy = (
            _first_nonempty(env, "NO_PROXY", "no_proxy") or DEFAULT_NO_PROXY
        )
        return cls(
            http_proxy=_validate_proxy_url(http_proxy),
            https_proxy=_validate_proxy_url(https_proxy),
            no_proxy=no_proxy,
        )

    def as_environment(self) -> dict[str, str]:
        """Return synchronized proxy variables for a GitHub-only child tool."""

        return {
            "HTTP_PROXY": self.http_proxy,
            "HTTPS_PROXY": self.https_proxy,
            "http_proxy": self.http_proxy,
            "https_proxy": self.https_proxy,
            "NO_PROXY": self.no_proxy,
            "no_proxy": self.no_proxy,
        }


def apply_proxy_environment(
    *,
    proxy_url: str | None = None,
    environment: MutableMapping[str, str] | None = None,
) -> ProxySettings:
    """Normalize proxy variables for a GitHub-only subprocess environment."""

    target = os.environ if environment is None else environment
    settings = ProxySettings.from_environment(target, proxy_url=proxy_url)
    target.update(settings.as_environment())
    return settings


@dataclass(frozen=True, slots=True)
class ProxyCheckResult:
    proxy_url: str
    host: str
    port: int


def check_proxy_connectivity(
    settings: ProxySettings | None = None,
    *,
    timeout_seconds: float = 3.0,
    protocol: str = "https",
) -> ProxyCheckResult:
    """Open a TCP connection to the selected proxy.

    This intentionally checks the proxy endpoint rather than an arbitrary
    public URL.  It fails quickly and does not consume/download remote content.
    Provider authentication and artifact availability are checked by the
    actual request afterwards.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    active = settings or ProxySettings.from_environment()
    if protocol not in {"http", "https"}:
        raise ValueError("protocol must be 'http' or 'https'")
    proxy_url = active.https_proxy if protocol == "https" else active.http_proxy
    if not proxy_url:
        raise ProxyConfigurationError("proxy connectivity check requested without a proxy")
    parsed = urlparse(proxy_url)
    host = parsed.hostname
    if host is None:  # guarded by ProxySettings, retained for type narrowing
        raise ProxyConfigurationError("proxy URL has no hostname")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        connection = socket.create_connection((host, port), timeout=timeout_seconds)
        connection.close()
    except OSError as error:
        raise ProxyConnectivityError(
            f"cannot reach proxy {host}:{port} within {timeout_seconds:g}s: {error}"
        ) from error
    return ProxyCheckResult(proxy_url=proxy_url, host=host, port=port)

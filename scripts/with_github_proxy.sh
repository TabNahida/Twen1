#!/usr/bin/env bash
# Run a GitHub-only network command with Twen's normalized proxy settings.
set -Eeuo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: scripts/with_github_proxy.sh COMMAND [ARG ...]" >&2
  exit 64
fi

default_proxy="http://172.23.240.1:8080"
if [[ -n "${TWEN_PROXY_URL:-}" ]]; then
  http_url="${TWEN_PROXY_URL}"
  https_url="${TWEN_PROXY_URL}"
else
  http_url="${HTTP_PROXY:-${http_proxy:-${default_proxy}}}"
  https_url="${HTTPS_PROXY:-${https_proxy:-${http_url}}}"
fi

export HTTP_PROXY="${http_url}"
export HTTPS_PROXY="${https_url}"
export http_proxy="${http_url}"
export https_proxy="${https_url}"
export NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1}}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

if [[ "${TWEN_SKIP_PROXY_CHECK:-0}" != "1" ]]; then
  timeout_seconds="${TWEN_PROXY_CHECK_TIMEOUT:-3}"
  python3 -c '
import socket
import sys
from urllib.parse import urlparse

url = sys.argv[1]
timeout = float(sys.argv[2])
parsed = urlparse(url)
if parsed.scheme not in {"http", "https"} or not parsed.hostname:
    raise SystemExit(f"invalid HTTP proxy URL: {url!r}")
port = parsed.port or (443 if parsed.scheme == "https" else 80)
try:
    connection = socket.create_connection((parsed.hostname, port), timeout=timeout)
    connection.close()
except OSError as error:
    raise SystemExit(
        f"proxy connectivity check failed for {parsed.hostname}:{port}: {error}"
    )
' "${HTTPS_PROXY}" "${timeout_seconds}"
fi

exec "$@"

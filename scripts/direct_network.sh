#!/usr/bin/env bash
# Run a non-GitHub network command with every conventional proxy disabled.
set -Eeuo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: scripts/direct_network.sh COMMAND [ARG ...]" >&2
  exit 64
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
exec "$@"

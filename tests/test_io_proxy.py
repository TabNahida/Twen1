from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from twen.io.proxy import (
    DEFAULT_NO_PROXY,
    DEFAULT_PROXY_URL,
    ProxySettings,
    apply_proxy_environment,
    check_proxy_connectivity,
)


class ProxySettingsTest(unittest.TestCase):
    def test_defaults_populate_both_cases_and_protocols(self) -> None:
        environment: dict[str, str] = {}
        settings = apply_proxy_environment(environment=environment)
        self.assertEqual(settings.http_proxy, DEFAULT_PROXY_URL)
        self.assertEqual(settings.https_proxy, DEFAULT_PROXY_URL)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            self.assertEqual(environment[name], DEFAULT_PROXY_URL)
        self.assertEqual(environment["NO_PROXY"], DEFAULT_NO_PROXY)
        self.assertEqual(environment["no_proxy"], DEFAULT_NO_PROXY)

    def test_common_override_wins(self) -> None:
        environment = {"HTTP_PROXY": "http://old.invalid:1"}
        settings = apply_proxy_environment(
            proxy_url="http://proxy.example:9000", environment=environment
        )
        self.assertEqual(settings.http_proxy, "http://proxy.example:9000")
        self.assertEqual(environment["https_proxy"], "http://proxy.example:9000")

    @patch("twen.io.proxy.socket.create_connection")
    def test_connectivity_check_only_opens_proxy_socket(self, create: Mock) -> None:
        connection = Mock()
        create.return_value = connection
        result = check_proxy_connectivity(
            ProxySettings("http://host.example:8123", "http://host.example:8123", ""),
            timeout_seconds=1.25,
        )
        create.assert_called_once_with(("host.example", 8123), timeout=1.25)
        connection.close.assert_called_once_with()
        self.assertEqual(result.port, 8123)

    def test_wrapper_forwards_arguments_without_eval(self) -> None:
        script = Path(__file__).parents[1] / "scripts" / "with_github_proxy.sh"
        code = (
            "import json,os,sys; "
            "print(json.dumps({'argv':sys.argv[1:],'proxy':os.environ['HTTP_PROXY'],"
            "'lower':os.environ['https_proxy']}))"
        )
        environment = {
            "PATH": os.environ["PATH"],
            "TWEN_SKIP_PROXY_CHECK": "1",
            "TWEN_PROXY_URL": "http://override.example:7777",
        }
        result = subprocess.run(
            [str(script), "python3", "-c", code, "a b", "; touch never"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["argv"], ["a b", "; touch never"])
        self.assertEqual(payload["proxy"], "http://override.example:7777")
        self.assertEqual(payload["lower"], "http://override.example:7777")

    def test_direct_wrapper_removes_all_proxy_variables(self) -> None:
        script = Path(__file__).parents[1] / "scripts" / "direct_network.sh"
        code = (
            "import json,os; "
            "print(json.dumps({name:os.environ.get(name) for name in "
            "['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']}))"
        )
        environment = {
            "PATH": os.environ["PATH"],
            "HTTP_PROXY": "http://ambient.example:1",
            "HTTPS_PROXY": "http://ambient.example:2",
            "ALL_PROXY": "socks5://ambient.example:3",
            "http_proxy": "http://ambient.example:4",
            "https_proxy": "http://ambient.example:5",
            "all_proxy": "socks5://ambient.example:6",
        }
        result = subprocess.run(
            [str(script), "python3", "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertTrue(all(value is None for value in json.loads(result.stdout).values()))


if __name__ == "__main__":
    unittest.main()

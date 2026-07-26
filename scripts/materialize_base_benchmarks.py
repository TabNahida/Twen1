#!/usr/bin/env python3
"""Lock and materialize the Base-v2 13-gram benchmark registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from twen.data.audits import inspect_benchmark_registry
from twen.data.benchmarks import materialize_base_benchmarks
from twen.io.download import DownloadManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve pinned public benchmark Parquet identities, download them with "
            "Range-resume/SHA verification, and emit deterministic audit JSONL."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("locks/base-benchmark-registry.json"),
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("data/benchmarks/base-v2"),
    )
    parser.add_argument(
        "--network-policy",
        choices=("direct", "github-only", "fallback", "proxy"),
        default="fallback",
        help="fallback tries Hugging Face directly before the configured proxy",
    )
    parser.add_argument(
        "--endpoint",
        default="https://huggingface.co",
        help="Hugging Face endpoint; source revisions remain immutable commits",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="optional Hugging Face token (defaults to HF_TOKEN)",
    )
    parser.add_argument(
        "--refresh-source-lock",
        action="store_true",
        help="re-query pinned metadata and prove it equals the existing native source lock",
    )
    parser.add_argument(
        "--rebuild-outputs",
        action="store_true",
        help=(
            "re-project every locked source; authenticate old JSONL against the registry "
            "before any atomic converter migration"
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform no network/write; verify ready JSONL identities and print the report",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--lock-timeout-seconds", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_only:
        report = inspect_benchmark_registry(
            args.registry,
            benchmark_root=args.benchmark_root,
            verify_hashes=True,
        )
    else:
        manager = DownloadManager(
            network_policy=args.network_policy,
            request_timeout_seconds=args.request_timeout_seconds,
            lock_timeout_seconds=args.lock_timeout_seconds,
        )
        report = materialize_base_benchmarks(
            args.registry,
            args.benchmark_root,
            manager=manager,
            token=args.token,
            endpoint=args.endpoint,
            refresh_source_lock=args.refresh_source_lock,
            rebuild_outputs=args.rebuild_outputs,
            lock_timeout_seconds=args.lock_timeout_seconds,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate that one CUDA compiler and its headers form a coherent toolkit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-home", default="/usr/local/cuda")
    parser.add_argument("--arch", default="sm_120a")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="also compile a header-only cubin; no GPU or runtime execution is used",
    )
    parser.add_argument("--output", default=None)
    return parser


def _first_existing(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"none of the required files exists: {[str(path) for path in paths]}")


def _macro(path: Path, name: str) -> int:
    match = re.search(
        rf"^\s*#\s*define\s+{re.escape(name)}\s+(\d+)",
        path.read_text(encoding="utf-8", errors="replace"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"{name} is absent from {path}")
    return int(match.group(1))


def _nvcc_release(nvcc: Path) -> tuple[str, str]:
    output = subprocess.run(
        [str(nvcc), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    release = re.search(r"release\s+(\d+\.\d+)", output)
    build = re.search(r"\bV(\d+\.\d+\.\d+)\b", output)
    if release is None or build is None:
        raise ValueError(f"cannot parse nvcc version output: {output!r}")
    return release.group(1), build.group(1)


def inspect_toolchain(cuda_home: str | Path) -> dict[str, Any]:
    home = Path(cuda_home).expanduser().resolve()
    nvcc = home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise FileNotFoundError(nvcc)
    runtime_header = _first_existing(
        (
            home / "include" / "cuda_runtime_api.h",
            home / "targets" / "x86_64-linux" / "include" / "cuda_runtime_api.h",
        )
    )
    cccl_header = _first_existing(
        (
            home / "include" / "cccl" / "cuda" / "std" / "__cccl" / "version.h",
            home
            / "targets"
            / "x86_64-linux"
            / "include"
            / "cccl"
            / "cuda"
            / "std"
            / "__cccl"
            / "version.h",
        )
    )
    nvcc_release, nvcc_build = _nvcc_release(nvcc)
    cudart_raw = _macro(runtime_header, "CUDART_VERSION")
    cudart_release = f"{cudart_raw // 1000}.{(cudart_raw % 1000) // 10}"
    cccl_raw = _macro(cccl_header, "CCCL_VERSION")
    cccl_release = f"{cccl_raw // 1_000_000}.{(cccl_raw // 1000) % 1000}.{cccl_raw % 1000}"
    return {
        "cuda_home": str(home),
        "nvcc": str(nvcc),
        "nvcc_release": nvcc_release,
        "nvcc_build": nvcc_build,
        "cudart_header": str(runtime_header),
        "cudart_release": cudart_release,
        "cccl_header": str(cccl_header),
        "cccl_release": cccl_release,
        "compiler_headers_match": nvcc_release == cudart_release,
    }


def compile_header_smoke(report: dict[str, Any], *, arch: str) -> dict[str, Any]:
    if not re.fullmatch(r"sm_\d+[a-z]?", arch):
        raise ValueError(f"invalid CUDA architecture: {arch!r}")
    source = """
#include <cuda_runtime.h>
#include <cuda/atomic>
extern "C" __global__ void twen_cuda_header_smoke(int* output) {
  cuda::atomic_ref<int, cuda::thread_scope_device> value(*output);
  value.fetch_add(1, cuda::memory_order_relaxed);
}
""".strip()
    with tempfile.TemporaryDirectory(prefix="twen-cuda-toolchain-") as raw_directory:
        directory = Path(raw_directory)
        source_path = directory / "header_smoke.cu"
        cubin_path = directory / "header_smoke.cubin"
        source_path.write_text(source + "\n", encoding="utf-8")
        command = [
            report["nvcc"],
            "--cubin",
            "-O2",
            f"-arch={arch}",
            "-std=c++17",
            str(source_path),
            "-o",
            str(cubin_path),
        ]
        started = __import__("time").perf_counter()
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        seconds = __import__("time").perf_counter() - started
        return {
            "requested": True,
            "ok": process.returncode == 0 and cubin_path.is_file(),
            "arch": arch,
            "seconds": seconds,
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "cubin_bytes": cubin_path.stat().st_size if cubin_path.is_file() else 0,
        }


def _write_result(result: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = inspect_toolchain(args.cuda_home)
        compile_report = (
            compile_header_smoke(report, arch=args.arch)
            if args.compile
            else {"requested": False, "ok": None, "arch": args.arch}
        )
        result = {
            "ok": bool(
                report["compiler_headers_match"] and (not args.compile or compile_report["ok"])
            ),
            "toolchain": report,
            "compile_smoke": compile_report,
        }
    except Exception as error:  # Keep environment failures machine-readable.
        result = {
            "ok": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "requested_cuda_home": args.cuda_home,
        }
    _write_result(result, args.output)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
